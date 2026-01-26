"""
🎯 CONFIG - Wspólna konfiguracja dla MiniGPT-60M
"""

import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== KONFIGURACJA SYSTEMU ====================
class SystemConfig:
    """Konfiguracja systemu i urządzeń"""

    def __init__(self):
        self.device = self._get_device()
        self.set_seeds(42)
        self._print_info()

    def _get_device(self) -> str:
        """Automatycznie wybiera najlepsze urządzenie"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def set_seeds(self, seed: int = 42):
        """Ustawia seed dla reprodukowalności"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _print_info(self):
        """Wyświetla informacje o systemie"""
        logger.info("=" * 60)
        logger.info("🎯 SYSTEM MINIGPT-60M")
        logger.info("=" * 60)
        logger.info(f"Python: {sys.version.split()[0]}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"Device: {self.device.upper()}")

        if self.device == "cuda":
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA dostępne: {torch.cuda.is_available()}")
            logger.info(f"Liczba GPU: {gpu_count}")
            for i in range(gpu_count):
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

        logger.info("=" * 60)

# ==================== KONFIGURACJA MODELU ====================
class ModelConfig:
    """Konfiguracja modelu 60M parametrów"""

    def __init__(self):
        # Słownik
        self.vocab_chars = list("aąbcćdeęfghijklłmnńoóprsśtuwyzźżAĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ")
        self.vocab_chars += list("0123456789")
        self.vocab_chars += list(" .,?!:;()[]{}+-*/=<>_\"'`~@#$%^&|\\/\n\t")
        self.vocab_chars += ["  ", "\n\n", "\t\t", "->", "::", "=>"]

        self.vocab = self.vocab_chars
        self.vocab_size = len(self.vocab)

        # Architektura dla ~60M parametrów
        self.embed_dim = 768
        self.n_layers = 12
        self.n_heads = 12
        self.max_len = 512
        self.ff_dim = self.embed_dim * 4
        self.dropout = 0.1
        self.activation = "gelu"
        self.norm_eps = 1e-5

        # Trening
        self.epochs = 3
        self.batch_size = 16 if torch.cuda.is_available() else 4
        self.grad_accum_steps = 4
        self.learning_rate = 3e-4
        self.weight_decay = 0.1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_eps = 1e-8
        self.clip_grad = 1.0
        self.warmup_steps = 2000

        # Mixed Precision
        self.use_amp = torch.cuda.is_available()

        # Parallel
        self.num_workers = 4 if torch.cuda.is_available() else 0
        self.pin_memory = True

        # Generowanie
        self.generation_temperature = 0.8
        self.top_k = 50
        self.top_p = 0.95
        self.repetition_penalty = 1.1

        # Ścieżki
        self.model_dir = "models"
        self.data_dir = "data"
        self.prepared_dir = "prepared_data"
        self.log_dir = "logs"
        self.tensorboard_dir = "runs"
        self.cache_dir = ".cache"
        self.checkpoints_dir = "checkpoints"
        self.resume_file = "resume_state.json"  # Plik stanu do wznowienia

        # Tworzenie katalogów
        self._create_dirs()

    def _create_dirs(self):
        """Tworzy wymagane katalogi"""
        dirs = [self.model_dir, self.data_dir, self.prepared_dir,
                self.log_dir, self.tensorboard_dir, self.cache_dir,
                "backups", "results", self.checkpoints_dir]

        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def print_config(self):
        """Wyświetla konfigurację"""
        logger.info("=" * 60)
        logger.info("⚙️ KONFIGURACJA MODELU")
        logger.info("=" * 60)
        logger.info(f"• Vocab size: {self.vocab_size}")
        logger.info(f"• Embed dim: {self.embed_dim}")
        logger.info(f"• Warstwy: {self.n_layers}")
        logger.info(f"• Głowy: {self.n_heads}")
        logger.info(f"• Kontekst: {self.max_len}")
        logger.info(f"• Batch size: {self.batch_size}")
        logger.info(f"• Learning rate: {self.learning_rate}")
        logger.info(f"• Mixed precision: {self.use_amp}")
        logger.info("=" * 60)

    def save_resume_state(self, state: Dict[str, Any]):
        """Zapisuje stan do wznowienia"""
        state_path = Path(self.checkpoints_dir) / self.resume_file
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Stan zapisany do {state_path}")

    def load_resume_state(self) -> Optional[Dict[str, Any]]:
        """Wczytuje stan do wznowienia"""
        state_path = Path(self.checkpoints_dir) / self.resume_file
        if state_path.exists():
            with open(state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Znajduje najnowszy checkpoint"""
        checkpoints = list(Path(self.checkpoints_dir).glob("checkpoint_*.pt"))
        if checkpoints:
            # Sortuj po czasie modyfikacji
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return checkpoints[0]
        return None

    def get_latest_model(self) -> Optional[Path]:
        """Znajduje najnowszy model"""
        models = list(Path(self.model_dir).glob("model_*.pt"))
        if models:
            # Szukaj model_final.pt, potem model_epoch_X.pt
            final_model = Path(self.model_dir) / "model_final.pt"
            if final_model.exists():
                return final_model

            # Sortuj po numerze epoki
            def get_epoch_num(path: Path) -> int:
                try:
                    # model_epoch_10.pt -> 10
                    name = path.stem
                    return int(name.split('_')[-1])
                except:
                    return 0

            models.sort(key=get_epoch_num, reverse=True)
            return models[0]
        return None

# Inicjalizacja konfiguracji
sys_config = SystemConfig()
cfg = ModelConfig()