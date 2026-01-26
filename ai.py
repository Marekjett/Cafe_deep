"""
🤖 AI - Model MiniGPT-60M i trening z checkpointami
"""

import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import shutil

from config import logger, cfg, sys_config

# ==================== TOKENIZER ====================
class AdvancedTokenizer:
    """Zaawansowany tokenizer"""

    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for i, c in enumerate(vocab)}

        # Cache
        self.encode_cache = {}
        self.pad_token = ' '
        self.pad_id = self.char2idx.get(self.pad_token, 0)
        self.eos_token = '\n'
        self.eos_id = self.char2idx.get(self.eos_token, 0)
        self.unk_token = '?'
        self.unk_id = self.char2idx.get(self.unk_token, 0)

    def encode(self, text: str, max_len: Optional[int] = None,
               truncation: bool = True, padding: bool = True) -> List[int]:
        """Kodowanie tekstu"""
        if max_len is None:
            max_len = cfg.max_len

        # Cache
        cache_key = f"{text[:50]}_{max_len}"
        if cache_key in self.encode_cache:
            return self.encode_cache[cache_key]

        ids = []
        i = 0
        text_len = len(text)

        while i < text_len:
            matched = False

            for token_len in range(4, 0, -1):
                if i + token_len <= text_len:
                    token = text[i:i + token_len]
                    if token in self.char2idx:
                        ids.append(self.char2idx[token])
                        i += token_len
                        matched = True
                        break

            if not matched:
                char = text[i]
                ids.append(self.char2idx.get(char, self.unk_id))
                i += 1

        # Przycinanie
        if truncation and len(ids) > max_len:
            keep_start = max_len // 2
            keep_end = max_len - keep_start
            ids = ids[:keep_start] + ids[-keep_end:]

        # Padding
        if padding and len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))

        self.encode_cache[cache_key] = ids
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Dekodowanie"""
        chars = []
        for token_id in ids:
            if skip_special and token_id == self.pad_id:
                continue
            if token_id < self.vocab_size:
                chars.append(self.idx2char[token_id])
            else:
                chars.append(self.unk_token)
        return ''.join(chars)

# Inicjalizacja tokenizera
tokenizer = AdvancedTokenizer(cfg.vocab)

# ==================== DATASET ====================
class TextDataset(Dataset):
    """Dataset z przygotowanych danych"""

    def __init__(self, data_file: str, max_samples: int = 100000):
        self.data_file = Path(data_file)
        self.samples = self._load_samples(max_samples)
        logger.info(f"📊 Dataset: {len(self.samples):,} próbek")

    def _load_samples(self, max_samples: int) -> List[str]:
        """Wczytuje próbki z pliku"""
        if not self.data_file.exists():
            logger.error(f"❌ Brak pliku z danymi: {self.data_file}")
            return self._create_dummy_data()

        samples = []
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                # Przetwarzaj linia po linii
                for i, line in enumerate(lines):
                    if i >= max_samples:
                        break

                    line = line.strip()
                    if line and 20 < len(line) < 5000:
                        samples.append(line)

                        if len(samples) % 10000 == 0:
                            logger.info(f"   Wczytano {len(samples):,} próbek...")

        except Exception as e:
            logger.error(f"❌ Błąd wczytywania danych: {e}")
            samples = self._create_dummy_data()

        return samples

    def _create_dummy_data(self) -> List[str]:
        """Tworzy przykładowe dane"""
        dummy = [
            "Python to język programowania wysokiego poziomu.",
            "Sieci neuronowe uczą się na danych.",
            "Adam Mickiewicz napisał Pana Tadeusza.",
            "Sztuczna inteligencja zmienia świat.",
            "Machine learning to poddziedzina AI."
        ]
        return dummy * 1000

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.samples[idx]

        # Tokenizacja
        ids = tokenizer.encode(text, cfg.max_len + 1, truncation=True, padding=True)

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        return x, y

# ==================== MODEL ====================
class MultiHeadAttention(nn.Module):
    """Multi-head attention"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projekcje
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Projekcje Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e4)

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Apply to values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Final projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        return output

class FeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Blok transformera"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x

class MiniGPT60M(nn.Module):
    """Główny model 60M parametrów"""

    def __init__(self):
        super().__init__()

        # Embeddingi
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embedding = nn.Embedding(cfg.max_len, cfg.embed_dim)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        # Bloki transformera
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.n_heads, cfg.ff_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # Final layers
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        # Inicjalizacja
        self._init_weights()

        # Oblicz parametry
        self._count_parameters()

    def _init_weights(self):
        """Inicjalizacja wag"""
        for module in self.modules():
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
        """Liczy parametry"""
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"🤖 Model: {total:,} parametrów ({total/1e6:.1f}M)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        # Przycinanie
        if seq_len > cfg.max_len:
            x = x[:, -cfg.max_len:]
            seq_len = cfg.max_len

        # Embeddings
        token_embeds = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.pos_embedding(positions)

        h = self.embed_dropout(token_embeds + pos_embeds)

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        # Final
        h = self.final_norm(h)
        logits = self.lm_head(h)

        return logits

    @torch.no_grad()
    def generate(self, prompt: str, max_len: int = 200, temperature: float = 0.8) -> str:
        """Generowanie tekstu"""
        self.eval()

        ids = tokenizer.encode(prompt, cfg.max_len)
        x = torch.tensor(ids).unsqueeze(0).to(sys_config.device)

        generated_ids = []

        for _ in range(max_len):
            # Przycinanie
            if x.size(1) > cfg.max_len:
                x = x[:, -cfg.max_len:]

            # Forward
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                logits = self(x)

            next_logits = logits[0, -1, :]

            # Temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Softmax i sampling
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            # Dodaj
            generated_ids.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], device=sys_config.device)], dim=1)

            # Stop na końcu zdania
            if next_id == tokenizer.eos_id and len(generated_ids) > 10:
                break

        # Decode
        all_ids = ids + generated_ids
        result = tokenizer.decode(all_ids)

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
                'max_len': cfg.max_len
            },
            'vocab': cfg.vocab,
            'metadata': metadata or {}
        }

        torch.save(checkpoint, path)
        logger.info(f"💾 Model zapisany do {path}")

    def save_checkpoint(self, path: str, optimizer, scheduler, epoch: int,
                       step: int, loss: float, metadata: Optional[Dict] = None):
        """Zapisuje pełny checkpoint (model + stan treningu)"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'config': {
                'vocab_size': cfg.vocab_size,
                'embed_dim': cfg.embed_dim,
                'n_layers': cfg.n_layers,
                'n_heads': cfg.n_heads,
                'max_len': cfg.max_len
            },
            'vocab': cfg.vocab,
            'metadata': metadata or {}
        }

        torch.save(checkpoint, path)
        logger.info(f"💾 Checkpoint zapisany do {path}")

    @classmethod
    def load(cls, path: str):
        """Wczytuje model"""
        checkpoint = torch.load(path, map_location=sys_config.device)
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ Model wczytany z {path}")
        return model

    @classmethod
    def load_checkpoint(cls, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """Wczytuje checkpoint (model + stan treningu)"""
        checkpoint = torch.load(path, map_location=sys_config.device)

        # Stwórz model
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])

        # Wczytaj stan optimizera i schedulera
        extra_info = {}
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            extra_info['optimizer_loaded'] = True

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            extra_info['scheduler_loaded'] = True

        logger.info(f"✅ Checkpoint wczytany z {path}")
        logger.info(f"   Epoka: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"   Krok: {checkpoint.get('step', 'N/A')}")
        logger.info(f"   Loss: {checkpoint.get('loss', 'N/A')}")

        return model, checkpoint.get('epoch', 0), checkpoint.get('step', 0), extra_info

# ==================== TRENER Z CHECKPOINTAMI ====================
class Trainer:
    """Trener modelu z checkpointami"""

    def __init__(self, model: MiniGPT60M, train_dataset: Dataset,
                 val_dataset: Optional[Dataset] = None,
                 start_epoch: int = 0, start_step: int = 0):
        self.model = model.to(sys_config.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )

        # Scheduler
        total_steps = len(train_dataset) * cfg.epochs // cfg.batch_size // cfg.grad_accum_steps
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Mixed precision
        self.scaler = GradScaler(enabled=cfg.use_amp)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=cfg.tensorboard_dir)

        # Statystyki
        self.stats = {
            'train_loss': [],
            'val_loss': [],
            'epoch': start_epoch,
            'step': start_step,
            'start_time': time.time(),
            'best_loss': float('inf')
        }

        # Checkpointing
        self.checkpoint_interval = 1000  # Co 1000 kroków
        self.last_checkpoint_step = start_step

        logger.info(f"🏋️ Trainer initialized (epoch: {start_epoch}, step: {start_step})")

    def train_epoch(self, epoch: int) -> float:
        """Trening jednej epoki"""
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

        pbar = tqdm(train_loader, desc=f"Epoka {epoch}")

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(sys_config.device, non_blocking=True)
            y = y.to(sys_config.device, non_blocking=True)

            # Mixed precision forward
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                logits = self.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, cfg.vocab_size),
                    y.view(-1),
                    ignore_index=tokenizer.pad_id
                )
                loss = loss / cfg.grad_accum_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Optimizer step
            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Update stats
                self.stats['step'] += 1

                # Zapisz checkpoint
                if self.stats['step'] - self.last_checkpoint_step >= self.checkpoint_interval:
                    self._save_checkpoint(epoch, loss.item() * cfg.grad_accum_steps)
                    self.last_checkpoint_step = self.stats['step']

                # Log
                if self.stats['step'] % 10 == 0:
                    self.writer.add_scalar('Train/loss', loss.item() * cfg.grad_accum_steps, self.stats['step'])
                    self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.stats['step'])

                # Update progress bar
                if batch_idx % 20 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item() * cfg.grad_accum_steps:.3f}",
                        'step': self.stats['step'],
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })

            epoch_loss += loss.item() * cfg.grad_accum_steps

            # Zapisz stan do wznowienia co 100 batchy
            if batch_idx % 100 == 0:
                self._save_resume_state(epoch, batch_idx)

        pbar.close()

        avg_loss = epoch_loss / len(train_loader)
        self.stats['train_loss'].append(avg_loss)
        self.stats['epoch'] = epoch

        return avg_loss

    def validate(self) -> float:
        """Walidacja"""
        if self.val_dataset is None:
            return float('inf')

        self.model.eval()
        total_loss = 0.0

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Walidacja", leave=False):
                x = x.to(sys_config.device, non_blocking=True)
                y = y.to(sys_config.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    logits = self.model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, cfg.vocab_size),
                        y.view(-1),
                        ignore_index=tokenizer.pad_id
                    )

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.stats['val_loss'].append(avg_loss)

        # Zapisz najlepszy model
        if avg_loss < self.stats['best_loss']:
            self.stats['best_loss'] = avg_loss
            self.model.save(f"{cfg.model_dir}/model_best.pt", {
                'epoch': self.stats['epoch'],
                'val_loss': avg_loss,
                'train_loss': self.stats['train_loss'][-1] if self.stats['train_loss'] else None
            })
            logger.info(f"🏆 Nowy najlepszy model! Val loss: {avg_loss:.4f}")

        return avg_loss

    def train(self, epochs: int = cfg.epochs, resume: bool = False):
        """Główna pętla treningu"""
        logger.info("=" * 60)
        logger.info("🚀 ROZPOCZĘCIE TRENINGU")
        if resume:
            logger.info("📈 WZNIOWANIE TRENINGU")
        logger.info("=" * 60)

        start_epoch = self.stats['epoch']
        start_time = time.time()

        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            logger.info(f"\n📈 EPOKA {epoch}/{start_epoch + epochs}")

            # Trening
            train_loss = self.train_epoch(epoch)
            logger.info(f"   Train loss: {train_loss:.4f}")

            # Walidacja
            if self.val_dataset:
                val_loss = self.validate()
                logger.info(f"   Val loss: {val_loss:.4f}")

            # Zapisz checkpoint epoki
            if epoch % 2 == 0 or epoch == start_epoch + epochs:
                self.model.save(f"{cfg.model_dir}/model_epoch_{epoch}.pt", {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss if self.val_dataset else None
                })

            # Generuj przykład
            if epoch % 3 == 0:
                self._generate_example(epoch)

            # Zapisz finalny checkpoint epoki
            self._save_checkpoint(epoch, train_loss, is_epoch_end=True)

        # Zapisz finalny model
        total_time = time.time() - start_time
        self.model.save(f"{cfg.model_dir}/model_final.pt", {
            'total_epochs': start_epoch + epochs,
            'total_time': total_time,
            'final_train_loss': self.stats['train_loss'][-1] if self.stats['train_loss'] else None,
            'best_val_loss': self.stats['best_loss']
        })

        # Usuń plik resume
        self._cleanup_resume_state()

        # Zamknij writer
        self.writer.close()

        # Podsumowanie
        self._print_summary(total_time)

    def train_more(self, additional_epochs: int = 3):
        """Kontynuuje trening istniejącego modelu"""
        logger.info(f"📈 DODATKOWY TRENING: {additional_epochs} epok")
        self.train(epochs=additional_epochs, resume=True)

    def _save_checkpoint(self, epoch: int, loss: float, is_epoch_end: bool = False):
        """Zapisuje checkpoint"""
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{self.stats['step']}.pt"
        if is_epoch_end:
            checkpoint_name = f"checkpoint_epoch_{epoch}_final.pt"

        checkpoint_path = Path(cfg.checkpoints_dir) / checkpoint_name

        self.model.save_checkpoint(
            str(checkpoint_path),
            self.optimizer,
            self.scheduler,
            epoch,
            self.stats['step'],
            loss,
            {
                'train_loss_history': self.stats['train_loss'],
                'val_loss_history': self.stats['val_loss'],
                'best_loss': self.stats['best_loss']
            }
        )

    def _save_resume_state(self, epoch: int, batch_idx: int):
        """Zapisuje stan do wznowienia"""
        resume_state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'step': self.stats['step'],
            'train_loss': self.stats['train_loss'][-1] if self.stats['train_loss'] else None,
            'optimizer_lr': self.optimizer.param_groups[0]['lr'],
            'timestamp': time.time()
        }

        cfg.save_resume_state(resume_state)

    def _cleanup_resume_state(self):
        """Czyści plik resume"""
        resume_path = Path(cfg.checkpoints_dir) / cfg.resume_file
        if resume_path.exists():
            resume_path.unlink()
            logger.info("🧹 Plik resume wyczyszczony")

    def _generate_example(self, epoch: int):
        """Generuje przykład"""
        self.model.eval()

        prompts = [
            "Python to",
            "Sztuczna inteligencja",
            "W przyszłości"
        ]

        logger.info(f"\n🎨 Przykłady (Epoka {epoch}):")

        for prompt in prompts:
            response = self.model.generate(prompt, max_len=100)
            logger.info(f"   '{prompt}' → '{response[:60]}...'")

        self.model.train()

    def _print_summary(self, total_time: float):
        """Wyświetla podsumowanie"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        logger.info("=" * 60)
        logger.info("🎉 TRENING ZAKOŃCZONY!")
        logger.info("=" * 60)
        logger.info(f"   • Czas: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logger.info(f"   • Epoki: {self.stats['epoch']}")
        logger.info(f"   • Kroki: {self.stats['step']}")

        if self.stats['train_loss']:
            logger.info(f"   • Final train loss: {self.stats['train_loss'][-1]:.4f}")

        if self.stats['val_loss']:
            logger.info(f"   • Best val loss: {self.stats['best_loss']:.4f}")

        logger.info(f"\n💾 Zapisane modele:")
        logger.info(f"   • {cfg.model_dir}/model_final.pt")
        logger.info(f"   • {cfg.model_dir}/model_best.pt")
        logger.info(f"   • {cfg.model_dir}/model_epoch_*.pt")
        logger.info(f"   • {cfg.checkpoints_dir}/checkpoint_*.pt")

        logger.info(f"\n🎮 KONTYNUACJA:")
        logger.info(f"   python main.py --cont    # Wznów trening")
        logger.info(f"   python main.py --more    # Dodatkowy trening")
        logger.info("=" * 60)

# ==================== FUNKCJE POMOCNICZE ====================
def create_train_val_split(dataset: Dataset, val_ratio: float = 0.1) -> Tuple[Dataset, Dataset]:
    """Dzieli dataset na treningowy i walidacyjny"""
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"📊 Podział danych: {train_size} trening, {val_size} walidacja")
    return train_dataset, val_dataset

def load_latest_checkpoint(model: Optional[MiniGPT60M] = None,
                          optimizer: Optional[torch.optim.Optimizer] = None,
                          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
    """Wczytuje najnowszy checkpoint"""
    checkpoint_path = cfg.get_latest_checkpoint()

    if checkpoint_path:
        logger.info(f"🔄 Wczytuję checkpoint: {checkpoint_path.name}")

        if model is None:
            model = MiniGPT60M()

        model, epoch, step, extra_info = MiniGPT60M.load_checkpoint(
            str(checkpoint_path), optimizer, scheduler
        )

        return model, epoch, step, extra_info
    else:
        logger.warning("⚠️ Nie znaleziono checkpointów")
        return None, 0, 0, {}

def load_resume_state() -> Optional[Dict]:
    """Wczytuje stan do wznowienia"""
    return cfg.load_resume_state()

# ==================== GŁÓWNE FUNKCJE AI ====================
def train_model(resume: bool = False, additional_epochs: int = 0):
    """Funkcja treningu modelu"""
    # Wczytaj dane
    data_file = Path(cfg.prepared_dir) / "all_data.txt"

    if not data_file.exists():
        logger.error(f"❌ Brak przygotowanych danych: {data_file}")
        logger.info("💡 Uruchom: python main.py --prepare")
        return

    dataset = TextDataset(str(data_file), max_samples=100000)

    if len(dataset) == 0:
        logger.error("❌ Brak danych do treningu!")
        return

    # Podziel na trening i walidację
    train_dataset, val_dataset = create_train_val_split(dataset, val_ratio=0.1)

    # Przygotuj model
    model = None
    start_epoch = 0
    start_step = 0

    if resume:
        # Spróbuj wczytać checkpoint
        model, start_epoch, start_step, _ = load_latest_checkpoint()

        if model is None:
            # Spróbuj wczytać ostatni model
            model_path = cfg.get_latest_model()
            if model_path:
                logger.info(f"🔄 Wczytuję ostatni model: {model_path.name}")
                model = MiniGPT60M.load(str(model_path))
                start_epoch = 0  # Zacznij od nowa, ale z wytrenowanymi wagami
            else:
                logger.info("🆕 Tworzę nowy model")
                model = MiniGPT60M()
    else:
        model = MiniGPT60M()

    # Trener
    trainer = Trainer(model, train_dataset, val_dataset, start_epoch, start_step)

    # Uruchom trening
    if additional_epochs > 0:
        trainer.train_more(additional_epochs)
    else:
        trainer.train(epochs=cfg.epochs, resume=resume)

def continue_training():
    """Kontynuuje trening od ostatniego checkpointu"""
    logger.info("🔄 KONTYNUACJA TRENINGU")

    # Sprawdź czy są checkpointy
    if not cfg.get_latest_checkpoint():
        logger.warning("⚠️ Nie znaleziono checkpointów, zaczynam od początku")
        train_model(resume=False)
        return

    # Kontynuuj trening
    train_model(resume=True)

def train_more_epochs(additional_epochs: int = 3):
    """Dodaje więcej epok treningu do istniejącego modelu"""
    logger.info(f"📈 DODAJĘ {additional_epochs} EPOK TRENINGU")

    # Użyj resume=True aby wczytać istniejący model
    train_model(resume=True, additional_epochs=additional_epochs)

def generate_text(prompt: str, model_path: Optional[str] = None):
    """Generuje tekst"""
    model = MiniGPT60M()

    # Spróbuj wczytać model
    if model_path and os.path.exists(model_path):
        model = MiniGPT60M.load(model_path)
    else:
        # Spróbuj wczytać ostatni model
        latest_model = cfg.get_latest_model()
        if latest_model:
            model = MiniGPT60M.load(str(latest_model))
        else:
            logger.warning("⚠️ Brak wytrenowanego modelu, używam niewytrenowanego")

    response = model.generate(prompt, max_len=300)

    print("\n" + "=" * 60)
    print("🎨 WYGENEROWANY TEKST")
    print("=" * 60)
    print(response)
    print("=" * 60)

def start_chat(model_path: Optional[str] = None):
    """Uruchamia czat"""
    model = MiniGPT60M()

    # Spróbuj wczytać model
    if model_path and os.path.exists(model_path):
        model = MiniGPT60M.load(model_path)
    else:
        # Spróbuj wczytać ostatni model
        latest_model = cfg.get_latest_model()
        if latest_model:
            model = MiniGPT60M.load(str(latest_model))
        else:
            logger.warning("⚠️ Brak wytrenowanego modelu, używam niewytrenowanego")

    chatbot = ChatBot(model)
    chatbot.chat()

# ==================== CHATBOT ====================
class ChatBot:
    """Interfejs czatu"""

    def __init__(self, model: MiniGPT60M):
        self.model = model
        self.model.eval()
        self.history = []

        logger.info("🤖 ChatBot initialized")

    def chat(self):
        """Rozmowa z użytkownikiem"""
        print("\n" + "=" * 60)
        print("🤖 MINIGPT-60M CHAT")
        print("=" * 60)
        print("Komendy: exit, clear, save, model, help")
        print("=" * 60)
        print()

        while True:
            try:
                user_input = input("\n🧑 Ty: ").strip()

                # Komendy
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Do widzenia!")
                    break

                elif user_input.lower() == 'clear':
                    self.history = []
                    print("🗑️ Historia wyczyszczona")
                    continue

                elif user_input.lower() == 'save':
                    self._save_conversation()
                    continue

                elif user_input.lower() == 'model':
                    self._show_model_info()
                    continue

                elif user_input.lower() == 'help':
                    self._show_help()
                    continue

                print("🤖 AI: ", end="", flush=True)

                # Generuj odpowiedź
                start_time = time.time()
                response = self.model.generate(user_input, max_len=200)
                gen_time = time.time() - start_time

                # Efekt pisania
                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(0.01)

                # Zapisz w historii
                self.history.append({
                    'user': user_input,
                    'ai': response,
                    'time': time.time()
                })

                print(f"\n   ⚡ {gen_time:.2f}s, {len(response.split())} słów")

            except KeyboardInterrupt:
                print("\n\n👋 Przerwano")
                break
            except Exception as e:
                print(f"\n❌ Błąd: {e}")

    def _save_conversation(self):
        """Zapisuje rozmowę"""
        if not self.history:
            print("⚠️ Brak historii do zapisania")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Rozmowa z MiniGPT-60M\n")
            f.write(f"Czas: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

            for i, exchange in enumerate(self.history, 1):
                f.write(f"{i}. Ty: {exchange['user']}\n")
                f.write(f"   AI: {exchange['ai']}\n\n")

        print(f"💾 Rozmowa zapisana do {filename}")

    def _show_model_info(self):
        """Pokazuje informacje o modelu"""
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n📊 INFORMACJE O MODELU:")
        print(f"   • Parametry: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"   • Warstwy: {cfg.n_layers}")
        print(f"   • Embed dim: {cfg.embed_dim}")
        print(f"   • Vocab size: {cfg.vocab_size}")

    def _show_help(self):
        """Pokazuje pomoc"""
        print("\n🆘 POMOC:")
        print("   • exit/quit/q - wyjście")
        print("   • clear - wyczyść historię")
        print("   • save - zapisz rozmowę do pliku")
        print("   • model - pokaż informacje o modelu")
        print("   • help - pokaż tę pomoc")

if __name__ == "__main__":
    # Przykładowe uruchomienie
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Trening modelu")
    parser.add_argument("--cont", action="store_true", help="Kontynuuj trening")
    parser.add_argument("--more", type=int, help="Dodaj N epok treningu", nargs='?', const=3)
    parser.add_argument("--generate", type=str, help="Generuj tekst")
    parser.add_argument("--chat", action="store_true", help="Tryb rozmowy")
    parser.add_argument("--model", type=str, help="Ścieżka do modelu")
    parser.add_argument("--epochs", type=int, default=cfg.epochs, help="Liczba epok")

    args = parser.parse_args()

    # Update liczby epok
    if args.epochs != cfg.epochs:
        cfg.epochs = args.epochs
        logger.info(f"⚙️ Ustawiono {cfg.epochs} epok")

    if args.train:
        train_model(resume=False)
    elif args.cont:
        continue_training()
    elif args.more:
        train_more_epochs(additional_epochs=args.more)
    elif args.generate:
        generate_text(args.generate, args.model)
    elif args.chat:
        start_chat(args.model)
    else:
        print("Użyj: python ai.py --train / --cont / --more [N] / --generate 'prompt' / --chat")