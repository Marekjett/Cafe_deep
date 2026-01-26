"""
🎯 MiniGPT-60M: Zaawansowany model językowy ~60 milionów parametrów
Autor: AI Assistant | Wersja: 4.0 Professional
"""

import os
import sys
import gc
import time
import math
import random
import json
import re
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    LambdaLR
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==================== KONFIGURACJA SYSTEMU ====================
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeviceConfig:
    """Konfiguracja urządzenia i pamięci"""

    def __init__(self):
        self.device = self._get_device()
        self.set_seeds(42)
        self._print_device_info()

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

    def _print_device_info(self):
        """Wyświetla informacje o urządzeniu"""
        logger.info(f"🎯 SYSTEM: Python {sys.version}")
        logger.info(f"🎯 PyTorch: {torch.__version__}")

        if self.device == "cuda":
            gpu_count = torch.cuda.device_count()
            logger.info(f"🎯 CUDA dostępne: {torch.cuda.is_available()}")
            logger.info(f"🎯 Liczba GPU: {gpu_count}")
            for i in range(gpu_count):
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"🎯 GPU {i}: {torch.cuda.get_device_name(i)} ({mem_total:.1f} GB)")

        elif self.device == "mps":
            logger.info("🎯 MPS (Apple Silicon) dostępne")

        logger.info(f"🎯 Wybrane urządzenie: {self.device.upper()}")


class ModelConfig:
    """Konfiguracja modelu 60M parametrów"""

    def __init__(self):
        # Słownik (rozszerzony o polskie znaki i symbole programistyczne)
        self.vocab_chars = list("aąbcćdeęfghijklłmnńoóprsśtuwyzźżAĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ")
        self.vocab_chars += list("0123456789")
        self.vocab_chars += list(" .,?!:;()[]{}+-*/=<>_\"'`~@#$%^&|\\/\n\t")
        self.vocab_chars += ["  ", "   ", "\n\n", "\t\t", "->", "::", "=>", "++", "--", "**", "//", "%%"]
        self.vocab_chars += ["def", "class", "import", "from", "return", "if", "else", "for", "while", "in", "is"]

        self.vocab = self.vocab_chars
        self.vocab_size = len(self.vocab)

        # Architektura dla ~60M parametrów
        self.embed_dim = 768  # Zwiększone dla 60M
        self.n_layers = 12  # 12 warstw
        self.n_heads = 12  # 12 głów
        self.max_len = 512  # Dłuższy kontekst
        self.ff_dim = self.embed_dim * 4
        self.dropout = 0.1
        self.activation = "gelu"
        self.norm_eps = 1e-5

        # Trening - DODAJ epochs TUTAJ:
        self.epochs = 3  # <-- DODAJ TUTAJ, nie na końcu!
        self.batch_size = 16 if torch.cuda.is_available() else 4
        self.grad_accum_steps = 4
        self.learning_rate = 3e-4
        self.weight_decay = 0.1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_eps = 1e-8
        self.clip_grad = 1.0
        self.warmup_steps = 2000

        # Harmonogram
        self.scheduler_type = "cosine"  # cosine, linear, plateau, onecycle
        self.lr_decay = 0.1
        self.min_lr = 1e-6

        # Mixed Precision
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Parallel
        self.data_parallel = False
        self.num_workers = 4 if torch.cuda.is_available() else 0
        self.pin_memory = True

        # Generowanie
        self.generation_temperature = 0.8
        self.top_k = 50
        self.top_p = 0.95
        self.repetition_penalty = 1.1
        self.beam_width = 3

        # Ścieżki
        self.model_dir = "models"
        self.data_dir = "data"
        self.prepared_dir = "prepared_data"
        self.log_dir = "logs"
        self.tensorboard_dir = "runs"

        # Tworzenie katalogów
        self._create_dirs()

        # USUŃ CAŁĄ RESZTĘ PONIŻEJ! (zduplikowany kod):
        # self.epochs = 5  # BRAKOWAŁO! - już dodane wyżej
        # self.epochs = 3  # Możesz ustawić mniej na początek - już jest 3
        # self.use_amp = torch.cuda.is_available() - już jest wyżej
        # self.num_workers = 4 if torch.cuda.is_available() else 0 - już jest wyżej
        # self.pin_memory = True - już jest wyżej

    def _create_dirs(self):
        """Tworzy wymagane katalogi"""
        dirs = [self.model_dir, self.data_dir, self.prepared_dir,
                self.log_dir, self.tensorboard_dir, "backups", "results"]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

# Inicjalizacja konfiguracji
device_cfg = DeviceConfig()
cfg = ModelConfig()


# ==================== ZAAWANSOWANY TOKENIZER ====================
class AdvancedTokenizer:
    """Zaawansowany tokenizer z cache'owaniem i statystykami"""

    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for i, c in enumerate(vocab)}

        # Cache dla szybkości
        self.encode_cache = {}
        self.statistics = {
            "total_tokens": 0,
            "total_chars": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Specjalne tokeny
        self.pad_token = ' '
        self.pad_id = self.char2idx.get(self.pad_token, 0)
        self.eos_token = '\n'
        self.eos_id = self.char2idx.get(self.eos_token, 0)
        self.unk_token = '?'
        self.unk_id = self.char2idx.get(self.unk_token, 0)

        logger.info(f"📊 Tokenizer zainicjalizowany: {self.vocab_size} tokenów")

    def encode(self, text: str, max_len: Optional[int] = None,
               truncation: bool = True, padding: bool = True) -> List[int]:
        """Zaawansowane kodowanie z cache'owaniem"""
        if max_len is None:
            max_len = cfg.max_len

        # Sprawdź cache
        cache_key = f"{text[:50]}_{max_len}_{truncation}_{padding}"
        if cache_key in self.encode_cache:
            self.statistics["cache_hits"] += 1
            return self.encode_cache[cache_key]

        self.statistics["cache_misses"] += 1
        self.statistics["total_chars"] += len(text)

        ids = []
        i = 0
        text_len = len(text)

        # Dopasowanie wieloznakowych tokenów
        while i < text_len:
            matched = False

            # Spróbuj najpierw dłuższe tokeny (do 4 znaków)
            for token_len in range(4, 0, -1):
                if i + token_len <= text_len:
                    token = text[i:i + token_len]
                    if token in self.char2idx:
                        ids.append(self.char2idx[token])
                        i += token_len
                        matched = True
                        break

            if not matched:
                # Użyj znaku lub UNK
                char = text[i]
                ids.append(self.char2idx.get(char, self.unk_id))
                i += 1

        # Przycinanie
        if truncation and len(ids) > max_len:
            # Zachowaj początek i koniec (dla kontekstu)
            keep_start = max_len // 2
            keep_end = max_len - keep_start
            ids = ids[:keep_start] + ids[-keep_end:]

        # Padding
        if padding and len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))

        self.statistics["total_tokens"] += len(ids)
        self.encode_cache[cache_key] = ids

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Dekodowanie z obsługą specjalnych tokenów"""
        chars = []
        for token_id in ids:
            if skip_special and token_id == self.pad_id:
                continue
            if token_id < self.vocab_size:
                chars.append(self.idx2char[token_id])
            else:
                chars.append(self.unk_token)
        return ''.join(chars)

    def encode_batch(self, texts: List[str], **kwargs) -> torch.Tensor:
        """Kodowanie batcha"""
        encoded = [self.encode(text, **kwargs) for text in texts]
        return torch.tensor(encoded, dtype=torch.long)

    def get_stats(self) -> Dict:
        """Zwraca statystyki tokenizacji"""
        return self.statistics.copy()

    def save(self, path: str):
        """Zapisuje tokenizer"""
        data = {
            'vocab': self.vocab,
            'char2idx': self.char2idx,
            'idx2char': self.idx2char,
            'statistics': self.statistics,
            'special_tokens': {
                'pad': self.pad_token,
                'eos': self.eos_token,
                'unk': self.unk_token
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Tokenizer zapisany do {path}")

    @classmethod
    def load(cls, path: str) -> 'AdvancedTokenizer':
        """Wczytuje tokenizer"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(data['vocab'])
        tokenizer.char2idx = data['char2idx']
        tokenizer.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        tokenizer.statistics = data['statistics']
        return tokenizer


tokenizer = AdvancedTokenizer(cfg.vocab)


# ==================== ZAAWANSOWANY DATASET ====================
class TextAugmentation:
    """Augmentacja danych tekstowych"""

    @staticmethod
    def random_deletion(text: str, p: float = 0.1) -> str:
        """Usuwa losowe słowa"""
        words = text.split()
        if len(words) == 1:
            return text

        kept_words = [word for word in words if random.random() > p]
        if len(kept_words) == 0:
            kept_words = [random.choice(words)]

        return ' '.join(kept_words)

    @staticmethod
    def random_swap(text: str, n: int = 3) -> str:
        """Zamienia losowe słowa miejscami"""
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)

    @staticmethod
    def random_insertion(text: str, n: int = 2) -> str:
        """Wstawia losowe słowa"""
        words = text.split()
        if len(words) == 0:
            return text

        for _ in range(n):
            word = random.choice(words)
            idx = random.randint(0, len(words))
            words.insert(idx, word)

        return ' '.join(words)

    @staticmethod
    def synonym_replacement(text: str, n: int = 2) -> str:
        """Zastępuje słowa synonimami (prosta implementacja)"""
        synonyms = {
            'dobry': ['świetny', 'wspaniały', 'znakomity'],
            'zły': ['kiepski', 'słaby', 'niedobry'],
            'duży': ['wielki', 'ogromny', 'spory'],
            'mały': ['drobny', 'niewielki', 'malutki'],
            'szybko': ['prędko', 'błyskawicznie', 'ekspresowo'],
            'wolno': ['powoli', 'ospale', 'leniwie']
        }

        words = text.split()
        new_words = words.copy()

        for _ in range(n):
            if not words:
                break

            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx].lower()

            if word in synonyms:
                synonym = random.choice(synonyms[word])
                new_words[word_idx] = synonym

        return ' '.join(new_words)


class SmartTextDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train",
                 augment: bool = True, cache: bool = True):
        # ZMIANA: Używaj bezpośrednio data_dir dla train
        if split == "train":
            self.data_dir = Path(data_dir)  # Bezpośrednio data/
        else:
            self.data_dir = Path(data_dir) / split  # Dla val/test

        self.split = split
        self.augment = augment and split == "train"
        self.cache = cache
        self.cached_samples = {}

        # Reszta kodu bez zmian...
    def _load_samples(self) -> List[str]:
        """Wczytuje próbki z plików"""
        samples = []

        for file_path in self.files:
            try:
                if file_path.suffix == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            samples.extend([str(item) for item in data])
                        elif isinstance(data, dict):
                            samples.extend([f"{k}: {v}" for k, v in data.items()])
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        chunks = re.split(r'\n\s*\n', content)
                        samples.extend([chunk.strip() for chunk in chunks if len(chunk.strip()) > 20])
            except Exception as e:
                logger.error(f"❌ Błąd wczytywania {file_path}: {e}")

        return samples

    def _create_dummy_data(self) -> List[str]:
        """Tworzy przykładowe dane jeśli brak"""
        logger.warning("⚠️ Tworzę przykładowe dane...")
        dummy_data = [
            "Python to język programowania wysokiego poziomu.",
            "Sieci neuronowe uczą się na danych.",
            "Adam Mickiewicz napisał Pana Tadeusza.",
            "Sztuczna inteligencja zmienia świat.",
            "def funkcja_przyklad(): return True",
            "Klasy i obiekty w programowaniu obiektowym.",
            "Polska literatura ma wielu wybitnych autorów.",
            "Machine learning to poddziedzina AI.",
            "Rekurencja to wywoływanie funkcji przez samą siebie.",
            "Walidacja krzyżowa poprawia generalizację modeli."
        ]
        return dummy_data * 10  # Powiel dla większej ilości

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = f"{self.split}_{idx}"

        if self.cache and cache_key in self.cached_samples:
            return self.cached_samples[cache_key]

        text = self.samples[idx]

        # Augmentacja dla danych treningowych
        if self.augment and random.random() > 0.5:
            augment_method = random.choice([
                TextAugmentation.random_deletion,
                TextAugmentation.random_swap,
                TextAugmentation.random_insertion,
                TextAugmentation.synonym_replacement
            ])
            text = augment_method(text)

        # Tokenizacja
        ids = tokenizer.encode(text, cfg.max_len + 1, truncation=True, padding=True)

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        if self.cache:
            self.cached_samples[cache_key] = (x, y)

        return x, y


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding - działająca wersja"""

    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        self.dim = dim

        # Oblicz częstotliwości - użyj float32 dla stabilności
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_len = max_len

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Tymczasowo wyłącz RoPE
        # q = self.rope(q, seq_len)
        # k = self.rope(k, seq_len)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask (causal mask dla autoregresji) - UŻYJ BEZPIECZNEJ FUNKCJI
        if mask is not None:
            attn_scores = masked_fill_fp16_safe(attn_scores, mask == 0, -1e4)
        else:
            # Causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            attn_scores = masked_fill_fp16_safe(attn_scores, causal_mask == 0, -1e4)

        # Reszta kodu bez zmian...
def masked_fill_fp16_safe(tensor: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
    """Safe masked_fill dla FP16"""
    if tensor.dtype == torch.float16:
        # Dla FP16 używaj mniejszych wartości
        if value < -1000:
            value = -1000.0
    return tensor.masked_fill(mask, value)
class MultiHeadAttention(nn.Module):
    """Multi-head attention z RoPE i kilkoma optymalizacjami"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim musi być podzielne przez num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projekcje Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Rotary positional embedding - WYŁĄCZONE TYMCZASOWO dla debugowania
        # self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Tymczasowo wyłącz RoPE
        # q = self.rope(q, seq_len)
        # k = self.rope(k, seq_len)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask (causal mask dla autoregresji)
        if mask is not None:
            # Użyj -1e4 zamiast -1e9 dla mixed precision
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)
        else:
            # Causal mask - użyj -1e4 dla FP16
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, -1e4)

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Apply to values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Final projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        return output
class GatedFeedForward(nn.Module):
    """Gated Feed Forward Network (lepsza niż standardowa)"""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()

        self.gate_proj = nn.Linear(embed_dim, ff_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Wybór funkcji aktywacji
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Nieznana funkcja aktywacji: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated mechanism (jak w LLaMA)
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)

        hidden = gate * up
        hidden = self.dropout(hidden)

        output = self.down_proj(hidden)
        output = self.dropout(output)

        return output


class TransformerBlock(nn.Module):
    """Zaawansowany blok transformera z pre-normalizacją"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()

        # Pre-norm (bardziej stabilne)
        self.input_norm = nn.LayerNorm(embed_dim, eps=cfg.norm_eps)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.post_attn_norm = nn.LayerNorm(embed_dim, eps=cfg.norm_eps)
        self.ffn = GatedFeedForward(embed_dim, ff_dim, dropout, activation)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Inicjalizacja (ważne dla głębokich sieci)
        self._init_weights()

    def _init_weights(self):
        """Dokładna inicjalizacja wag"""
        # Xavier/Glorot dla warstw liniowych
        for module in [self.attn.q_proj, self.attn.k_proj, self.attn.v_proj, self.attn.out_proj,
                       self.ffn.gate_proj, self.ffn.up_proj, self.ffn.down_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention z residual connection
        attn_input = self.input_norm(x)
        attn_output = self.attn(attn_input)
        x = x + self.dropout(attn_output)

        # Feed-forward z residual connection
        ffn_input = self.post_attn_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.dropout(ffn_output)

        return x


class MiniGPT60M(nn.Module):
    """Główny model ~60M parametrów"""

    def __init__(self):
        super().__init__()

        # Embeddingi
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embedding = nn.Embedding(cfg.max_len, cfg.embed_dim)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        # Bloki transformera
        self.blocks = nn.ModuleList([
            TransformerBlock(
                cfg.embed_dim,
                cfg.n_heads,
                cfg.ff_dim,
                cfg.dropout,
                cfg.activation
            ) for _ in range(cfg.n_layers)
        ])

        # Final layers
        self.final_norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # Tie weights (embedding i output)
        self.lm_head.weight = self.token_embedding.weight

        # Inicjalizacja
        self.apply(self._init_weights)

        # Oblicz parametry
        self._count_parameters()

        # DataParallel jeśli wiele GPU
        if torch.cuda.device_count() > 1 and cfg.data_parallel:
            self = nn.DataParallel(self)
            logger.info(f"🎯 Używam DataParallel na {torch.cuda.device_count()} GPU")

    def _init_weights(self, module):
        """Inicjalizacja wag dla różnych typów warstw"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _count_parameters(self):
        """Liczy i wyświetla liczbę parametrów"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Szczegółowy podział
        embed_params = sum(p.numel() for p in self.token_embedding.parameters())
        transformer_params = total_params - embed_params

        logger.info("=" * 60)
        logger.info("🤖 MODEL MINIGPT-60M")
        logger.info("=" * 60)
        logger.info(f"   • Całkowite parametry: {total_params:,} ({total_params / 1e6:.1f}M)")
        logger.info(f"   • Trainable: {trainable_params:,}")
        logger.info(f"   • Embedding: {embed_params:,}")
        logger.info(f"   • Transformer: {transformer_params:,}")
        logger.info(f"   • Embed dim: {cfg.embed_dim}")
        logger.info(f"   • Warstwy: {cfg.n_layers}")
        logger.info(f"   • Głowy: {cfg.n_heads}")
        logger.info(f"   • Kontekst: {cfg.max_len}")
        logger.info(f"   • Vocab size: {cfg.vocab_size}")
        logger.info("=" * 60)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        # Sprawdź długość sekwencji
        if seq_len > cfg.max_len:
            x = x[:, -cfg.max_len:]
            seq_len = cfg.max_len

        # Token embeddings
        token_embeds = self.token_embedding(x)

        # Position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.pos_embedding(positions)

        # Sum and dropout
        h = self.embed_dropout(token_embeds + pos_embeds)

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        # Final layer norm
        h = self.final_norm(h)

        # Language modeling head
        logits = self.lm_head(h)

        return logits

    @torch.no_grad()
    def generate(
            self,
            prompt: str,
            max_len: int = 200,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.1,
            beam_width: int = 1,
            stop_tokens: Optional[List[str]] = None
    ) -> str:
        """
        Zaawansowane generowanie tekstu z wieloma strategiami
        """
        self.eval()

        if stop_tokens is None:
            stop_tokens = ['.', '!', '?', '\n\n']

        # Beam search czy sampling?
        if beam_width > 1:
            return self._beam_search(prompt, max_len, beam_width, stop_tokens)

        # Standardowe sampling
        ids = tokenizer.encode(prompt, cfg.max_len)
        x = torch.tensor(ids).unsqueeze(0).to(device_cfg.device)

        generated_ids = []

        for step in range(max_len):
            # Przycinaj jeśli za długie
            if x.size(1) > cfg.max_len:
                x = x[:, -cfg.max_len:]

            # Forward pass
            logits = self(x)
            next_logits = logits[0, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    next_logits[token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[-1]] = -float('inf')

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = -float('inf')

            # Temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Softmax i sampling
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            # Dodaj do sekwencji
            generated_ids.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]]).to(device_cfg.device)], dim=1)

            # Warunki stopu
            char = tokenizer.idx2char.get(next_id, '')
            if char in stop_tokens and step > 10:
                if random.random() < 0.3:
                    break

        # Stwórz wynik
        all_ids = ids + generated_ids
        result = tokenizer.decode(all_ids)

        # Wyodrębnij tylko wygenerowaną część
        if result.startswith(prompt):
            result = result[len(prompt):].strip()

        return result

    def _beam_search(self, prompt: str, max_len: int, beam_width: int, stop_tokens: List[str]) -> str:
        """Implementacja beam search"""
        # Implementacja beam search (uproszczona)
        ids = tokenizer.encode(prompt, cfg.max_len)
        beams = [(torch.tensor(ids).unsqueeze(0).to(device_cfg.device), 0.0)]

        for step in range(max_len):
            new_beams = []

            for beam_tensor, beam_score in beams:
                # Forward pass
                logits = self(beam_tensor)
                next_logits = logits[0, -1, :]

                # Top-k candidates
                values, indices = torch.topk(F.log_softmax(next_logits, dim=-1), beam_width)

                for i in range(beam_width):
                    new_id = indices[i].item()
                    new_score = beam_score + values[i].item()

                    new_beam_tensor = torch.cat([
                        beam_tensor,
                        torch.tensor([[new_id]]).to(device_cfg.device)
                    ], dim=1)

                    # Przycinaj jeśli za długie
                    if new_beam_tensor.size(1) > cfg.max_len:
                        new_beam_tensor = new_beam_tensor[:, -cfg.max_len:]

                    new_beams.append((new_beam_tensor, new_score))

            # Wybierz najlepsze beam_width beamy
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Sprawdź warunki stopu
            last_token = tokenizer.idx2char.get(beams[0][0][0, -1].item(), '')
            if last_token in stop_tokens and step > 10:
                break

        # Zwróć najlepszy beam
        best_beam_ids = beams[0][0][0].tolist()
        result = tokenizer.decode(best_beam_ids)

        if result.startswith(prompt):
            result = result[len(prompt):].strip()

        return result

    def save(self, path: str, metadata: Optional[Dict] = None):
        """Zapisuje model z metadanymi"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': cfg.vocab_size,
                'embed_dim': cfg.embed_dim,
                'n_layers': cfg.n_layers,
                'n_heads': cfg.n_heads,
                'max_len': cfg.max_len,
                'dropout': cfg.dropout,
                'activation': cfg.activation
            },
            'tokenizer': tokenizer.vocab,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, path)
        logger.info(f"💾 Model zapisany do {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> 'MiniGPT60M':
        """Wczytuje model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model nie znaleziony: {path}")

        checkpoint = torch.load(path, map_location=device_cfg.device)

        # Stwórz model z konfiguracji checkpointa
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"✅ Model wczytany z {path}")
        logger.info(f"   Konfiguracja: {checkpoint['config']}")

        if 'metadata' in checkpoint:
            logger.info(f"   Metadata: {checkpoint['metadata'].get('description', 'Brak')}")

        return model


# ==================== SYSTEM OCENY LOSS_LESS ====================
class LossLessMetrics:
    """Zaawansowane metryki oceny modelu"""

    def __init__(self):
        self.history = []
        self.best_score = 100.0  # Mniej = lepiej

        # Kategorie pytań
        self.categories = {
            "polski": ["język", "gramatyka", "literatura"],
            "python": ["kod", "programowanie", "funkcje"],
            "nauka": ["AI", "sieci", "matematyka"],
            "ogólne": ["zdania", "logika", "kreatywność"]
        }

        # Wagi kategorii
        self.category_weights = {
            "polski": 0.3,
            "python": 0.4,
            "nauka": 0.2,
            "ogólne": 0.1
        }

    def evaluate(
            self,
            model: MiniGPT60M,
            questions: List[Dict],
            temperature: float = 0.8,
            verbose: bool = True
    ) -> Dict:
        """Pełna ocena modelu"""
        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info("🎯 SYSTEM OCENY LOSS_LESS")
            logger.info("=" * 60)

        model.eval()
        results = {
            "total_score": 0,
            "category_scores": defaultdict(float),
            "detailed": [],
            "timestamp": datetime.now().isoformat()
        }

        category_counts = defaultdict(int)

        for i, q in enumerate(tqdm(questions, desc="Ocenianie", disable=not verbose)):
            category = q.get("category", "ogólne")
            question = q["question"]
            expected = q.get("expected_keywords", [])
            perfect = q.get("perfect_answer", "")

            # Generuj odpowiedź
            response = model.generate(
                prompt=question,
                max_len=200,
                temperature=temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty
            )

            # Oceń odpowiedź
            score, metrics = self._score_single(
                question=question,
                response=response,
                expected_keywords=expected,
                perfect_answer=perfect
            )

            # Zbierz statystyki
            results["total_score"] += score
            results["category_scores"][category] += score
            category_counts[category] += 1

            results["detailed"].append({
                "question": question,
                "response": response,
                "score": score,
                "metrics": metrics,
                "category": category
            })

        model.train()

        # Oblicz średnie
        num_questions = len(questions)
        if num_questions > 0:
            results["avg_score"] = results["total_score"] / num_questions

            # Normalizuj do 0-10
            normalized_avg = min(10.0, results["avg_score"])

            # Loss_Less score (0-100, mniej = lepiej)
            loss_less = 100 - (normalized_avg * 10)

            # Średnie per kategoria
            for category in results["category_scores"]:
                if category_counts[category] > 0:
                    results["category_scores"][category] /= category_counts[category]

        else:
            loss_less = 100

        results["loss_less"] = loss_less

        # Zapisz do historii
        self.history.append({
            "timestamp": results["timestamp"],
            "loss_less": loss_less,
            "avg_score": results.get("avg_score", 0),
            "category_scores": dict(results["category_scores"])
        })

        # Aktualizuj najlepszy wynik
        if loss_less < self.best_score:
            self.best_score = loss_less

        if verbose:
            self._print_results(results)

        return results

    def _score_single(self, question: str, response: str,
                      expected_keywords: List[str], perfect_answer: str) -> Tuple[float, Dict]:
        """Ocenia pojedynczą odpowiedź"""
        metrics = {}

        # 1. Dopasowanie słów kluczowych (0-4 punkty)
        keyword_score = 0
        response_lower = response.lower()

        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                keyword_score += 1

        keyword_score = min(4, keyword_score)
        metrics["keyword_score"] = keyword_score

        # 2. Spójność i gramatyka (0-3 punkty)
        coherence_score = 0

        # Sprawdź czy są zdania
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if len(valid_sentences) >= 2:
            coherence_score += 1

        # Sprawdź wielkie litery na początku zdań
        if re.search(r'[.!?]\s+[A-ZĄĆĘŁŃÓŚŹŻ]', response):
            coherence_score += 1

        # Sprawdź czy odpowiedź nie jest za krótka
        if len(response.split()) >= 5:
            coherence_score += 1

        metrics["coherence_score"] = coherence_score

        # 3. Relewancja (0-3 punkty)
        relevance_score = 0

        # Usuń stop words i policz wspólne słowa
        stop_words = {"się", "i", "w", "z", "na", "do", "że", "aby", "ten", "ta", "to"}
        question_words = set([w.lower() for w in question.split() if w.lower() not in stop_words])
        response_words = set([w.lower() for w in response.split() if w.lower() not in stop_words])

        common_words = question_words.intersection(response_words)
        if len(common_words) >= 2:
            relevance_score += 1

        # Sprawdź czy odpowiedź bezpośrednio odpowiada na pytanie
        if any(word in response_lower for word in ["tak", "nie", "dlatego", "ponieważ"]):
            relevance_score += 1

        # Długość odpowiedzi (nie za krótka, nie za długa)
        word_count = len(response.split())
        if 10 <= word_count <= 100:
            relevance_score += 1

        metrics["relevance_score"] = relevance_score

        # Całkowity wynik (0-10)
        total_score = keyword_score + coherence_score + relevance_score
        total_score = min(10, total_score)

        return total_score, metrics

    def _print_results(self, results: Dict):
        """Wyświetla wyniki oceny"""
        logger.info("\n📊 WYNIKI OCENY:")
        logger.info("-" * 40)
        logger.info(f"   Loss_Less Score: {results['loss_less']:.1f}/100")
        logger.info(f"   Średni wynik: {results.get('avg_score', 0):.1f}/10")

        logger.info("\n   Wyniki per kategoria:")
        for category, score in results["category_scores"].items():
            logger.info(f"   • {category:10}: {score:.1f}/10")

        # Interpretacja
        ll_score = results["loss_less"]
        logger.info("\n   📈 INTERPRETACJA:")

        if ll_score <= 20:
            logger.info("   🎉 DOSKONAŁY! Model jest praktycznie idealny")
        elif ll_score <= 40:
            logger.info("   ✅ BARDZO DOBRY! Model jest bardzo użyteczny")
        elif ll_score <= 60:
            logger.info("   👍 DOBRY! Model jest funkcjonalny")
        elif ll_score <= 80:
            logger.info("   ⚠️  ŚREDNI! Model wymaga poprawy")
        else:
            logger.info("   ❌ SŁABY! Model potrzebuje dużo pracy")

        logger.info("=" * 60)

    def save_history(self, path: str = "loss_less_history.json"):
        """Zapisuje historię ocen"""
        data = {
            "history": self.history,
            "best_score": self.best_score,
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_evaluations": len(self.history),
                "categories": self.categories
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Historia Loss_Less zapisana do {path}")

    def plot_history(self, save_path: str = "loss_less_progress.png"):
        """Tworzy wykres postępu"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter

            if len(self.history) < 2:
                logger.warning("⚠️ Za mało danych do wykresu")
                return

            # Przygotuj dane
            timestamps = [datetime.fromisoformat(h["timestamp"]) for h in self.history]
            scores = [h["loss_less"] for h in self.history]

            # Stwórz wykres
            fig, ax = plt.subplots(figsize=(12, 6))

            # Linia główna
            ax.plot(timestamps, scores, 'b-o', linewidth=2, markersize=6, label='Loss_Less')

            # Wypełnienie obszarów
            ax.fill_between(timestamps, 0, 30, alpha=0.2, color='green', label='Doskonały')
            ax.fill_between(timestamps, 30, 60, alpha=0.2, color='yellow', label='Dobry')
            ax.fill_between(timestamps, 60, 80, alpha=0.2, color='orange', label='Średni')
            ax.fill_between(timestamps, 80, 100, alpha=0.2, color='red', label='Słaby')

            # Linia trendu
            if len(scores) >= 3:
                z = np.polyfit(range(len(scores)), scores, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(range(len(scores))), 'r--', alpha=0.7,
                        label=f'Trend: {"↓" if z[0] < 0 else "↑"} {abs(z[0]):.2f}/eval')

            # Konfiguracja
            ax.set_xlabel('Data')
            ax.set_ylabel('Loss_Less Score (mniej = lepiej)')
            ax.set_title('Postęp modelu - System Loss_Less')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            # Format daty
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Zapisz
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            logger.info(f"📈 Wykres zapisany jako {save_path}")

            # Pokaż jeśli w trybie interaktywnym
            if 'DISPLAY' in os.environ:
                plt.show()
            else:
                plt.close()

        except ImportError:
            logger.warning("⚠️ Matplotlib nie zainstalowany - pomijam wykres")


# ==================== ZAAWANSOWANY TRENING ====================
class AdvancedTrainer:
    """Zaawansowany trainer z wieloma funkcjami"""

    def __init__(self, model: MiniGPT60M, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        self.model = model.to(device_cfg.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay
        )

        # Scheduler
        total_steps = len(train_dataset) * cfg.epochs // cfg.batch_size // cfg.grad_accum_steps
        self.scheduler = self._create_scheduler(total_steps)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=0.1  # Pomaga w generalizacji
        )

        # Mixed precision
        self.scaler = GradScaler() if cfg.use_amp else None

        # TensorBoard
        self.writer = SummaryWriter(log_dir=cfg.tensorboard_dir)

        # System oceny
        self.evaluator = LossLessMetrics()

        # Statystyki
        self.stats = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "best_loss": float('inf'),
            "best_loss_less": 100.0,
            "epoch": 0,
            "step": 0,
            "start_time": time.time()
        }

        # Backup
        self.backup_counter = 0

        logger.info("🏋️ Zaawansowany Trainer zainicjalizowany")
        logger.info(f"   • Dataset: {len(train_dataset)} próbek")
        logger.info(f"   • Batch size: {cfg.batch_size}")
        logger.info(f"   • Gradient accumulation: {cfg.grad_accum_steps}")
        logger.info(f"   • Scheduler: {cfg.scheduler_type}")
        logger.info(f"   • Mixed precision: {cfg.use_amp}")

    def _create_scheduler(self, total_steps: int):
        """Tworzy scheduler w zależności od konfiguracji"""
        if cfg.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=cfg.min_lr
            )
        elif cfg.scheduler_type == "cosine_warm":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps // 10,
                T_mult=1,
                eta_min=cfg.min_lr
            )
        elif cfg.scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=cfg.learning_rate,
                total_steps=total_steps,
                pct_start=0.1
            )
        elif cfg.scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                min_lr=cfg.min_lr
            )
        else:
            # Linear warmup
            def lr_lambda(step):
                if step < cfg.warmup_steps:
                    return float(step) / float(max(1, cfg.warmup_steps))
                return max(0.0, float(total_steps - step) / float(max(1, total_steps - cfg.warmup_steps)))

            return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, epoch: int) -> float:
        """Wykonuje jedną epokę treningu"""
        self.model.train()
        epoch_loss = 0.0

        # DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True
        )

        # Pasek postępu
        pbar = tqdm(train_loader, desc=f"Epoka {epoch}", leave=False)

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device_cfg.device), y.to(device_cfg.device)

            # Mixed precision forward
            if cfg.use_amp:
                with autocast():
                    logits = self.model(x)
                    loss = self.criterion(logits.view(-1, cfg.vocab_size), y.view(-1))
                    loss = loss / cfg.grad_accum_steps
            else:
                logits = self.model(x)
                loss = self.criterion(logits.view(-1, cfg.vocab_size), y.view(-1))
                loss = loss / cfg.grad_accum_steps

            # Backward z gradient accumulation
            if cfg.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Krok optymalizacji po akumulacji
            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                # Gradient clipping
                if cfg.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad)

                # Optimizer step
                if cfg.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step
                if cfg.scheduler_type != "plateau":
                    self.scheduler.step()

                self.optimizer.zero_grad()

                # Aktualizuj statystyki
                self.stats["step"] += 1
                current_lr = self.optimizer.param_groups[0]['lr']
                self.stats["learning_rates"].append(current_lr)

                # Zapisz do TensorBoard
                if self.stats["step"] % 10 == 0:
                    self.writer.add_scalar('Train/loss', loss.item() * cfg.grad_accum_steps, self.stats["step"])
                    self.writer.add_scalar('Train/lr', current_lr, self.stats["step"])

                # Logowanie
                if self.stats["step"] % 50 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item() * cfg.grad_accum_steps:.3f}",
                        'lr': f"{current_lr:.2e}",
                        'step': self.stats["step"]
                    })

                # Generuj przykłady
                if self.stats["step"] % 100 == 0:
                    self._log_examples(epoch)

                # Backup
                if time.time() - self.stats.get("last_backup", 0) > 300:  # Co 5 minut
                    self._create_backup()

            epoch_loss += loss.item() * cfg.grad_accum_steps

            # Czyszczenie pamięci
            del x, y, logits, loss
            if device_cfg.device == "cuda":
                torch.cuda.empty_cache()

        pbar.close()

        # Średni loss epoki
        avg_epoch_loss = epoch_loss / len(train_loader)
        self.stats["train_loss"].append(avg_epoch_loss)

        return avg_epoch_loss

    def validate(self) -> float:
        """Walidacja modelu"""
        if self.val_dataset is None:
            return float('inf')

        self.model.eval()
        total_loss = 0.0

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Walidacja", leave=False):
                x, y = x.to(device_cfg.device), y.to(device_cfg.device)

                logits = self.model(x)
                loss = self.criterion(logits.view(-1, cfg.vocab_size), y.view(-1))

                total_loss += loss.item()

        avg_val_loss = total_loss / len(val_loader)
        self.stats["val_loss"].append(avg_val_loss)

        # Scheduler step dla ReduceLROnPlateau
        if cfg.scheduler_type == "plateau":
            self.scheduler.step(avg_val_loss)

        self.writer.add_scalar('Val/loss', avg_val_loss, self.stats["epoch"])

        return avg_val_loss

    def _log_examples(self, epoch: int):
        """Loguje przykłady generacji"""
        self.model.eval()

        prompts = [
            "Python to",
            "Hej, jak się masz?",
            "Adam Mickiewicz",
            "def funkcja",
            "Sieci neuronowe",
            "Sztuczna inteligencja",
            "Raspberry Pi",
            "Klasy w Pythonie"
        ]

        examples = []

        with torch.no_grad():
            for prompt in prompts[:3]:  # Tylko 3 dla szybkości
                response = self.model.generate(
                    prompt,
                    max_len=80,
                    temperature=0.8,
                    top_k=40
                )
                examples.append(f"**{prompt}** → {response}")

        # Zapisz do TensorBoard
        examples_text = "\n\n".join(examples)
        self.writer.add_text(f'Examples/epoch_{epoch}', examples_text, self.stats["step"])

        # Wyświetl w konsoli
        logger.info(f"\n🎨 Przykłady generacji (krok {self.stats['step']}):")
        for example in examples:
            logger.info(f"   {example}")

        self.model.train()

    def _create_backup(self):
        """Tworzy backup modelu"""
        self.backup_counter += 1
        backup_path = f"backups/model_backup_{self.backup_counter}_{int(time.time())}.pt"

        self.model.save(backup_path, {
            "epoch": self.stats["epoch"],
            "step": self.stats["step"],
            "loss": self.stats["train_loss"][-1] if self.stats["train_loss"] else 0,
            "description": f"Backup #{self.backup_counter}"
        })

        self.stats["last_backup"] = time.time()
        logger.info(f"💾 Backup #{self.backup_counter} zapisany: {backup_path}")

    def train(self, epochs: int = cfg.epochs):
        """Główna pętla treningu"""
        logger.info("\n" + "=" * 60)
        logger.info("🚀 ROZPOCZĘCIE ZAAWANSOWANEGO TRENINGU")
        logger.info("=" * 60)

        # Początkowa ocena Loss_Less
        logger.info("\n📋 OCENA POCZĄTKOWA:")
        initial_questions = self._load_test_questions()
        initial_eval = self.evaluator.evaluate(self.model, initial_questions[:5])
        self.stats["initial_loss_less"] = initial_eval["loss_less"]

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            self.stats["epoch"] = epoch

            logger.info(f"\n{'=' * 60}")
            logger.info(f"📈 EPOKA {epoch}/{epochs}")
            logger.info(f"{'=' * 60}")

            # Trening
            train_loss = self.train_epoch(epoch)
            logger.info(f"   📊 Loss treningu: {train_loss:.4f}")

            # Walidacja
            if self.val_dataset:
                val_loss = self.validate()
                logger.info(f"   📊 Loss walidacji: {val_loss:.4f}")

                # Zapisz jeśli lepszy loss
                if val_loss < self.stats["best_loss"]:
                    self.stats["best_loss"] = val_loss
                    self.model.save("model_best_loss.pt", {
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "type": "best_loss"
                    })
                    logger.info(f"   🏆 NAJLEPSZY model (loss) zapisany!")

            # Ocena Loss_Less co 2 epoki
            if epoch % 2 == 0 or epoch == epochs:
                logger.info(f"\n   🔍 Ocena Loss_Less (epoka {epoch}):")
                eval_results = self.evaluator.evaluate(self.model, initial_questions, verbose=False)

                loss_less = eval_results["loss_less"]
                self.writer.add_scalar('Eval/loss_less', loss_less, epoch)

                # Zapisz jeśli lepszy Loss_Less
                if loss_less < self.stats["best_loss_less"]:
                    self.stats["best_loss_less"] = loss_less
                    self.model.save("model_best_ll.pt", {
                        "epoch": epoch,
                        "loss_less": loss_less,
                        "type": "best_loss_less"
                    })
                    logger.info(f"   🏆 NAJLEPSZY model (Loss_Less: {loss_less:.1f}) zapisany!")

            # Zapisz checkpoint epoki
            self.model.save(f"model_epoch_{epoch}.pt", {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss if self.val_dataset else None,
                "step": self.stats["step"]
            })

            logger.info(f"\n   ✅ Epoka {epoch} zakończona")
            logger.info(f"   ⏱️  Czas: {self._format_time(time.time() - start_time)}")

        # Końcowa ocena
        logger.info("\n📋 OCENA KOŃCOWA:")
        final_eval = self.evaluator.evaluate(self.model, initial_questions)
        self.stats["final_loss_less"] = final_eval["loss_less"]

        # Zapisz historię
        self.evaluator.save_history()
        self.evaluator.plot_history()

        # Zamknij TensorBoard
        self.writer.close()

        # Podsumowanie
        total_time = time.time() - start_time
        self._print_summary(total_time)

        # Zapisz finalny model
        self.model.save("model_final.pt", {
            "epochs": epochs,
            "final_loss_less": self.stats["final_loss_less"],
            "best_loss_less": self.stats["best_loss_less"],
            "total_time": total_time
        })

    def _load_test_questions(self) -> List[Dict]:
        """Ładuje pytania testowe"""
        questions_file = "test_questions.json"

        if os.path.exists(questions_file):
            with open(questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("questions", [])

        # Domyślne pytania
        return [
            {"question": "Hej, jak się masz?", "expected_keywords": ["dobrze", "dziękuję"], "category": "polski"},
            {"question": "Jak napisać funkcję w Pythonie?", "expected_keywords": ["def", "return"],
             "category": "python"},
            {"question": "Kto napisał Pana Tadeusza?", "expected_keywords": ["Mickiewicz", "Adam"],
             "category": "literatura"}
        ]

    def _format_time(self, seconds: float) -> str:
        """Formatuje czas"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def _print_summary(self, total_time: float):
        """Wyświetla podsumowanie treningu"""
        logger.info("\n" + "=" * 60)
        logger.info("🎉 TRENING ZAKOŃCZONY!")
        logger.info("=" * 60)

        logger.info(f"\n📊 PODSUMOWANIE:")
        logger.info(f"   • Czas treningu: {self._format_time(total_time)}")
        logger.info(f"   • Epoki: {self.stats['epoch']}")
        logger.info(f"   • Kroki: {self.stats['step']}")

        if "initial_loss_less" in self.stats and "final_loss_less" in self.stats:
            improvement = self.stats["initial_loss_less"] - self.stats["final_loss_less"]
            logger.info(f"   • Loss_Less początkowy: {self.stats['initial_loss_less']:.1f}")
            logger.info(f"   • Loss_Less końcowy: {self.stats['final_loss_less']:.1f}")
            logger.info(f"   • Poprawa: {improvement:+.1f} punktów")

        logger.info(f"   • Najlepszy Loss_Less: {self.stats.get('best_loss_less', 100):.1f}")

        logger.info(f"\n💾 ZAPISANE MODELE:")
        logger.info(f"   • model_best_loss.pt - najlepszy loss walidacyjny")
        logger.info(f"   • model_best_ll.pt - najlepszy Loss_Less")
        logger.info(f"   • model_final.pt - finalny model")
        logger.info(f"   • model_epoch_*.pt - checkpoints epok")

        logger.info(f"\n📈 WIZUALIZACJE:")
        logger.info(f"   • TensorBoard: tensorboard --logdir={cfg.tensorboard_dir}")
        logger.info(f"   • Wykres Loss_Less: loss_less_progress.png")
        logger.info(f"   • Historia ocen: loss_less_history.json")

        logger.info(f"\n🎮 PRZETESTOJ MODEL:")
        logger.info(f"   python main.py --talk")
        logger.info(f"   python main.py --evaluate")


# ==================== INTERFEJS UŻYTKOWNIKA ====================
class InteractiveChat:
    """Zaawansowany interfejs czatu"""

    def __init__(self, model_path: str = "model_final.pt"):
        self.model = MiniGPT60M.load(model_path) if os.path.exists(model_path) else MiniGPT60M()
        self.model.eval()

        self.history = []
        self.max_history = 20
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"\n💬 ZAAWANSOWANY INTERFEJS CZATU")
        logger.info("=" * 60)
        logger.info(f"   Model: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parametrów")
        logger.info(f"   Device: {device_cfg.device.upper()}")
        logger.info(f"   ID rozmowy: {self.conversation_id}")
        logger.info("\n   Polecenia:")
        logger.info("   • exit, quit - wyjście")
        logger.info("   • clear - wyczyść historię")
        logger.info("   • save - zapisz rozmowę")
        logger.info("   • temp X.X - ustaw temperaturę (0.1-2.0)")
        logger.info("   • len XXX - ustaw długość odpowiedzi (10-1000)")
        logger.info("   • history - pokaż historię")
        logger.info("   • config - pokaż konfigurację")
        logger.info("-" * 60)

    def chat(self):
        """Główna pętla rozmowy"""
        print("\n🤖 Witaj! Rozmawiaj ze mną. Wpisz 'exit' aby zakończyć.\n")

        # Parametry
        temperature = 0.8
        max_length = 200

        while True:
            try:
                # Input użytkownika
                user_input = input("\n🧑 Ty: ").strip()

                # Komendy specjalne
                if self._handle_command(user_input, temperature, max_length):
                    continue

                # Dodaj do historii
                self.history.append(f"U: {user_input}")
                if len(self.history) > self.max_history * 2:
                    self.history = self.history[-self.max_history * 2:]

                # Przygotuj kontekst
                context = self._build_context()

                # Generuj odpowiedź
                print("🤖 AI: ", end="", flush=True)

                start_time = time.time()
                response = self.model.generate(
                    prompt=context,
                    max_len=max_length,
                    temperature=temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                    repetition_penalty=cfg.repetition_penalty
                )
                gen_time = time.time() - start_time

                # Wyświetl z efektem pisania
                self._typewriter_effect(response)

                # Dodaj odpowiedź do historii
                self.history.append(f"A: {response}")

                # Statystyki
                print(f"\n   ⚡ Wygenerowano w {gen_time:.2f}s, {len(response.split())} słów")

            except KeyboardInterrupt:
                print("\n\n⚠️ Przerwano przez użytkownika")
                if input("💾 Zapisać rozmowę? (t/n): ").lower().startswith('t'):
                    self.save_conversation()
                break

            except Exception as e:
                print(f"\n❌ Błąd: {e}")
                import traceback
                traceback.print_exc()

    def _handle_command(self, command: str, temp: float, max_len: int) -> bool:
        """Obsługuje komendy specjalne"""
        cmd = command.lower().strip()

        if cmd in ['exit', 'quit', 'q', 'wyjdz']:
            print("\n👋 Do widzenia!")
            if input("💾 Zapisać rozmowę? (t/n): ").lower().startswith('t'):
                self.save_conversation()
            sys.exit(0)

        elif cmd in ['clear', 'czysc']:
            self.history = []
            print("🗑️ Historia wyczyszczona")
            return True

        elif cmd in ['save', 'zapisz']:
            self.save_conversation()
            return True

        elif cmd in ['history', 'historia']:
            self.show_history()
            return True

        elif cmd in ['config', 'konfig']:
            self.show_config()
            return True

        elif cmd.startswith('temp '):
            try:
                new_temp = float(cmd.split()[1])
                if 0.1 <= new_temp <= 2.0:
                    temp = new_temp
                    print(f"🌡️ Temperatura ustawiona na {temp}")
                else:
                    print("❌ Temperatura musi być między 0.1 a 2.0")
            except:
                print("❌ Błędna temperatura, użyj: temp 0.8")
            return True

        elif cmd.startswith('len '):
            try:
                new_len = int(cmd.split()[1])
                if 10 <= new_len <= 1000:
                    max_len = new_len
                    print(f"📏 Długość ustawiona na {max_len}")
                else:
                    print("❌ Długość musi być między 10 a 1000")
            except:
                print("❌ Błędna długość, użyj: len 200")
            return True

        return False

    def _build_context(self) -> str:
        """Buduje kontekst z historii"""
        if not self.history:
            return ""

        # Ostatnie 6 wymian (12 wiadomości)
        recent_history = self.history[-12:] if len(self.history) > 12 else self.history
        return "\n".join(recent_history) + "\nA: "

    def _typewriter_effect(self, text: str, speed: float = 0.01):
        """Efekt pisania na maszynie"""
        for char in text:
            print(char, end="", flush=True)
            time.sleep(speed)

    def show_history(self):
        """Pokazuje historię rozmowy"""
        print("\n📜 Historia rozmowy:")
        print("-" * 50)
        for i, msg in enumerate(self.history[-10:], 1):
            prefix = "🧑" if msg.startswith("U:") else "🤖"
            print(f"{i:2}. {prefix} {msg[3:][:80]}{'...' if len(msg) > 80 else ''}")
        print("-" * 50)

    def show_config(self):
        """Pokazuje konfigurację"""
        print("\n⚙️ Konfiguracja modelu:")
        print("-" * 40)
        print(f"   • Parametry: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        print(f"   • Vocab size: {cfg.vocab_size}")
        print(f"   • Embed dim: {cfg.embed_dim}")
        print(f"   • Warstwy: {cfg.n_layers}")
        print(f"   • Głowy: {cfg.n_heads}")
        print(f"   • Kontekst: {cfg.max_len}")
        print("-" * 40)

    def save_conversation(self, filename: Optional[str] = None):
        """Zapisuje rozmowę do pliku"""
        if filename is None:
            filename = f"conversation_{self.conversation_id}.md"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Rozmowa z AI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Parametry\n")
            f.write(f"- Model: MiniGPT-60M\n")
            f.write(f"- Data: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"- ID: {self.conversation_id}\n\n")
            f.write("## Dialog\n\n")

            for msg in self.history:
                if msg.startswith("U:"):
                    f.write(f"### 🧑 Ty\n\n{msg[3:]}\n\n")
                else:
                    f.write(f"### 🤖 AI\n\n{msg[3:]}\n\n")

        print(f"💾 Rozmowa zapisana do {filename}")


# ==================== GŁÓWNA FUNKCJA ====================
def main():
    """Główna funkcja programu"""
    parser = argparse.ArgumentParser(
        description="🎯 MiniGPT-60M: Zaawansowany model językowy ~60M parametrów",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python main.py --train                    # Trening od zera
  python main.py --train --resume model.pt  # Kontynuacja treningu
  python main.py --talk                     # Rozmowa z modelem
  python main.py --evaluate                 # Ocena modelu (Loss_Less)
  python main.py --test                     # Testy jednostkowe
  python main.py --generate "Python to"     # Generuj tekst
  python main.py --config                   # Pokaż konfigurację
  python main.py --prepare-data            # Przygotuj dane
        """
    )

    parser.add_argument("--train", action="store_true", help="Trening modelu")
    parser.add_argument("--resume", type=str, help="Wznów trening z pliku")
    parser.add_argument("--talk", action="store_true", help="Tryb rozmowy")
    parser.add_argument("--evaluate", action="store_true", help="Oceń model systemem Loss_Less")
    parser.add_argument("--model", type=str, default="model_final.pt", help="Ścieżka do modelu")
    parser.add_argument("--test", action="store_true", help="Test modelu")
    parser.add_argument("--generate", type=str, help="Wygeneruj tekst z podanego promptu")
    parser.add_argument("--config", action="store_true", help="Pokaż konfigurację")
    parser.add_argument("--prepare-data", action="store_true", help="Przygotuj dane")
    parser.add_argument("--epochs", type=int, default=cfg.epochs, help="Liczba epok")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🎯 MINIGPT-60M - PROFESJONALNA WERSJA")
    logger.info("=" * 60)

    # Pokaż konfigurację
    if args.config:
        logger.info("\n⚙️ KONFIGURACJA SYSTEMU:")
        logger.info("-" * 40)
        logger.info(f"   • Device: {device_cfg.device.upper()}")
        logger.info(f"   • Vocab size: {cfg.vocab_size}")
        logger.info(f"   • Embed dim: {cfg.embed_dim}")
        logger.info(f"   • Warstwy: {cfg.n_layers}")
        logger.info(f"   • Głowy: {cfg.n_heads}")
        logger.info(f"   • Kontekst: {cfg.max_len}")
        logger.info(f"   • Parametry: ~{sum(p.numel() for p in MiniGPT60M().parameters()) / 1e6:.1f}M")
        logger.info(f"   • Batch size: {cfg.batch_size}")
        logger.info(f"   • Learning rate: {cfg.learning_rate}")
        logger.info("-" * 40)
        return

    # Przygotuj dane
    if args.prepare_data:
        from data_preparation import prepare_all_data
        prepare_all_data()
        return

    # Test jednostkowy
    if args.test:
        logger.info("\n🧪 TESTY JEDNOSTKOWE")
        logger.info("-" * 40)

        # Test modelu
        model = MiniGPT60M().to(device_cfg.device)

        # Test forward
        x = torch.randint(0, cfg.vocab_size, (2, 32)).to(device_cfg.device)
        y = torch.randint(0, cfg.vocab_size, (2, 32)).to(device_cfg.device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        logger.info(f"✅ Test forward: Loss = {loss.item():.4f}")

        # Test generacji
        gen = model.generate("Test", max_len=20)
        logger.info(f"✅ Test generacji: '{gen}'")

        # Test zapisu/odczytu
        test_path = "test_model.pt"
        model.save(test_path)
        model2 = MiniGPT60M.load(test_path)
        os.remove(test_path)
        logger.info("✅ Test zapisu/odczytu: PASS")

        # Test tokenizera
        text = "Test tokenizacji"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        logger.info(f"✅ Test tokenizera: '{text}' -> '{decoded}'")

        logger.info("\n🎯 WSZYSTKIE TESTY PRZESZŁY POMYŚLNIE!")
        return

    # Generuj tekst
    if args.generate:
        logger.info(f"\n🎨 GENEROWANIE TEKSTU: '{args.generate}'")

        model = MiniGPT60M()
        if os.path.exists(args.model):
            model = MiniGPT60M.load(args.model)

        response = model.generate(
            args.generate,
            max_len=300,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )

        logger.info("\n" + "=" * 60)
        logger.info("📝 WYJŚCIE:")
        logger.info("=" * 60)
        logger.info(response)
        logger.info("=" * 60)
        return

    # Ocena modelu
    if args.evaluate:
        logger.info("\n🎯 OCENA MODELU SYSTEMEM LOSS_LESS")

        model = MiniGPT60M()
        if os.path.exists(args.model):
            model = MiniGPT60M.load(args.model)

        evaluator = LossLessMetrics()

        # Załaduj pytania testowe
        questions_file = "test_questions.json"
        if os.path.exists(questions_file):
            with open(questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                questions = data.get("questions", [])
        else:
            questions = [
                {"question": "Hej, jak się masz?", "expected_keywords": ["dobrze", "dziękuję"], "category": "polski"},
                {"question": "Jak napisać funkcję w Pythonie?", "expected_keywords": ["def", "return"],
                 "category": "python"}
            ]

        results = evaluator.evaluate(model, questions)
        evaluator.save_history()
        evaluator.plot_history()

        logger.info(f"\n💾 Wyniki zapisane w loss_less_history.json")
        return

    # Trening
    if args.train:
        logger.info("\n🚀 ROZPOCZĘCIE TRENINGU")

        # Dataset
        train_dataset = SmartTextDataset(cfg.data_dir, split="train", augment=True)
        val_dataset = SmartTextDataset(cfg.data_dir, split="val", augment=False) \
            if os.path.exists(os.path.join(cfg.data_dir, "val")) else None

        if len(train_dataset) == 0:
            logger.error("❌ Brak danych treningowych!")
            logger.info("💡 Uruchom: python main.py --prepare-data")
            return

        # Model
        model = MiniGPT60M()

        # Wczytaj jeśli resume
        if args.resume and os.path.exists(args.resume):
            model = MiniGPT60M.load(args.resume)
            logger.info(f"✅ Wznawiam trening z {args.resume}")

        # Trainer
        trainer = AdvancedTrainer(model, train_dataset, val_dataset)

        # Uruchom trening
        trainer.train(epochs=args.epochs)
        return

    # Rozmowa
    if args.talk:
        chat = InteractiveChat(args.model)
        chat.chat()
        return

    # Jeśli żadna flaga, pokaż help
    logger.info("\n❓ Nie podano flagi. Dostępne opcje:\n")
    parser.print_help()


# ==================== URUCHOMIENIE ====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n👋 Program przerwany przez użytkownika")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Krytyczny błąd: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)