import json
import torch
import torch.nn as nn
import glob
import os
import random
import datetime
import numpy as np
import time
import shutil
import sys
import math
import re
from collections import defaultdict, Counter
from pathlib import Path

torch.set_num_threads(8)


# -------------------- Konfiguracja --------------------
class ChatbotConfig:
    def __init__(
            self,
            learn=True,
            talk=False,
            traintalk=False,
            train=False,
            force=False,
            epochs=2,
            batch=1,
            hidden=712,
            layers=8,
            lr=0.05,
            data="data/",
            books="book/",
            model_dir="model",
            backup_dir="backup_model",
            max_length=1024,
            dropout=0.2,
            backup_freq=5,
            main_save_freq=20,
            min_word_freq=5,
            context_size=100,
            train_talk_threshold=0.7,
            estimated_tokens_per_second=50000  # Szacowana wydajność (tokenów/sekundę)
    ):
        self.learn = learn
        self.talk = talk
        self.traintalk = traintalk
        self.train = train
        self.force = force
        self.epochs = epochs
        self.batch = batch
        self.hidden = hidden
        self.layers = layers
        self.lr = lr
        self.data = data
        self.books = books
        self.model_dir = model_dir
        self.backup_dir = backup_dir
        self.model_path = os.path.join(model_dir, "model.pt")
        self.config_path = os.path.join(model_dir, "config.json")
        self.max_length = max_length
        self.dropout = dropout
        self.backup_freq = backup_freq
        self.main_save_freq = main_save_freq
        self.min_word_freq = min_word_freq
        self.context_size = context_size
        self.train_talk_threshold = train_talk_threshold
        self.estimated_tokens_per_second = estimated_tokens_per_second

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.books, exist_ok=True)

    def save_config(self):
        try:
            config = {
                "epochs": self.epochs,
                "batch": self.batch,
                "hidden": self.hidden,
                "layers": self.layers,
                "lr": self.lr,
                "data": self.data,
                "books": self.books,
                "max_length": self.max_length,
                "dropout": self.dropout,
                "backup_freq": self.backup_freq,
                "main_save_freq": self.main_save_freq,
                "min_word_freq": self.min_word_freq,
                "context_size": self.context_size,
                "train_talk_threshold": self.train_talk_threshold,
                "estimated_tokens_per_second": self.estimated_tokens_per_second,
                "saved_at": datetime.datetime.now().isoformat()
            }
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"❌ Błąd zapisu konfiguracji: {e}")
            return False

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                for key, value in config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                return True
            except Exception as e:
                print(f"❌ Błąd wczytywania konfiguracji: {e}")
                return False
        return False


# -------------------- Pasek postępu --------------------
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = f"{100 * (iteration / float(total)):5.1f}%"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '░' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent} {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()


# -------------------- Preprocessing tekstu --------------------
class TextPreprocessor:
    @staticmethod
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()
        return text

    @staticmethod
    def split_into_chunks(text, chunk_size=1000):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    @staticmethod
    def prepare_conversation_line(input_text, output_text):
        return f"<BOS>{input_text.strip()}\n{output_text.strip()}<EOS>"


# -------------------- Obliczanie czasu treningu --------------------
def estimate_training_time(config, total_tokens):
    """
    Szacuje czas treningu na podstawie liczby tokenów

    Args:
        config: Konfiguracja modelu
        total_tokens: Całkowita liczba tokenów w datasetcie

    Returns:
        Tuple: (estimated_seconds, time_string)
    """
    # Liczba operacji na token
    # Dla LSTM: ~4 * hidden * layers operacji na token
    operations_per_token = 4 * config.hidden * config.layers

    # Całkowita liczba operacji
    total_operations = total_tokens * operations_per_token * config.epochs

    # Szacowana wydajność (operacji na sekundę)
    # Bazujemy na estimated_tokens_per_second z konfiguracji
    if torch.cuda.is_available():
        # Dla GPU: ~10-100x szybsze
        ops_per_second = config.estimated_tokens_per_second * operations_per_token * 0.5
    else:
        # Dla CPU
        ops_per_second = config.estimated_tokens_per_second * operations_per_token * 0.1

    if ops_per_second == 0:
        ops_per_second = 1e6  # Domyślna wartość

    estimated_seconds = total_operations / ops_per_second

    # Formatowanie czasu
    if estimated_seconds < 60:
        time_str = f"{estimated_seconds:.1f} sekund"
    elif estimated_seconds < 3600:
        minutes = estimated_seconds / 60
        time_str = f"{minutes:.1f} minut"
    elif estimated_seconds < 86400:
        hours = estimated_seconds / 3600
        time_str = f"{hours:.1f} godzin"
    else:
        days = estimated_seconds / 86400
        time_str = f"{days:.1f} dni"

    return estimated_seconds, time_str


# -------------------- Zapisywanie modeli --------------------
def save_checkpoint(config, checkpoint_data, epoch, val_loss, is_best=False, is_regular=False):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        files_saved = []
        save_messages = []

        if is_regular and epoch % config.backup_freq == 0:
            backup_path = os.path.join(config.backup_dir, f"model_epoch{epoch:04d}_{timestamp}.pt")
            print_progress_bar(0, 3, prefix=f"📁 Backup epoka {epoch}:", suffix="przygotowanie", length=30)
            time.sleep(0.1)

            torch.save(checkpoint_data, backup_path)
            print_progress_bar(1, 3, prefix=f"📁 Backup epoka {epoch}:", suffix="zapisywanie", length=30)
            time.sleep(0.1)

            if os.path.exists(backup_path):
                size = os.path.getsize(backup_path)
                print_progress_bar(3, 3, prefix=f"📁 Backup epoka {epoch}:",
                                   suffix=f"zapisano {size / 1024 / 1024:.1f} MB", length=30)
                save_messages.append(f"📁 Backup: model_epoch{epoch:04d}_{timestamp}.pt ({size / 1024 / 1024:.1f} MB)")
            else:
                print_progress_bar(3, 3, prefix=f"📁 Backup epoka {epoch}:", suffix="BŁĄD!", length=30)
                save_messages.append(f"❌ Backup: model_epoch{epoch:04d}_{timestamp}.pt (BŁĄD!)")

            files_saved.append(backup_path)

        if is_regular and epoch % config.main_save_freq == 0:
            print(f"\n{'=' * 60}")
            print(f"💾 NADPISYWANIE MODELU GŁÓWNEGO (epoch {epoch})...")
            print(f"{'=' * 60}")

            if os.path.exists(config.model_path):
                old_backup = os.path.join(config.backup_dir, f"model_main_old_{timestamp}.pt")
                try:
                    shutil.copy2(config.model_path, old_backup)
                    save_messages.append(f"📋 Kopia starego: model_main_old_{timestamp}.pt")
                except Exception as e:
                    save_messages.append(f"⚠️  Nie skopiowano starego: {e}")

            try:
                print_progress_bar(0, 3, prefix="💾 Model główny:", suffix="przygotowanie", length=30)
                time.sleep(0.1)

                torch.save(checkpoint_data, config.model_path)
                print_progress_bar(1, 3, prefix="💾 Model główny:", suffix="zapisywanie", length=30)
                time.sleep(0.1)

                config.save_config()
                print_progress_bar(2, 3, prefix="💾 Model główny:", suffix="konfiguracja", length=30)
                time.sleep(0.1)

                if os.path.exists(config.model_path):
                    size = os.path.getsize(config.model_path)
                    print_progress_bar(3, 3, prefix="💾 Model główny:",
                                       suffix=f"GOTOWE! {size / 1024 / 1024:.1f} MB", length=30)
                    save_messages.append(f"✅ MODEL GŁÓWNY: model.pt ({size / 1024 / 1024:.1f} MB)")
                else:
                    print_progress_bar(3, 3, prefix="💾 Model główny:", suffix="BŁĄD!", length=30)
                    save_messages.append(f"❌ MODEL GŁÓWNY: model.pt (BŁĄD!)")

                files_saved.append(config.model_path)
            except Exception as e:
                print(f"   ❌ Błąd zapisu modelu głównego: {e}")

        if is_best:
            best_path = os.path.join(config.model_dir, "model_best.pt")
            try:
                print_progress_bar(0, 2, prefix="🏆 Najlepszy model:", suffix="przygotowanie", length=30)
                time.sleep(0.1)

                torch.save(checkpoint_data, best_path)
                print_progress_bar(1, 2, prefix="🏆 Najlepszy model:", suffix="zapisywanie", length=30)
                time.sleep(0.1)

                if os.path.exists(best_path):
                    size = os.path.getsize(best_path)
                    print_progress_bar(2, 2, prefix="🏆 Najlepszy model:",
                                       suffix=f"GOTOWE! {size / 1024 / 1024:.1f} MB", length=30)
                    save_messages.append(f"🏆 NAJLEPSZY: model_best.pt ({size / 1024 / 1024:.1f} MB)")
                else:
                    print_progress_bar(2, 2, prefix="🏆 Najlepszy model:", suffix="BŁĄD!", length=30)
                    save_messages.append(f"❌ NAJLEPSZY: model_best.pt (BŁĄD!)")

                files_saved.append(best_path)
            except Exception as e:
                save_messages.append(f"⚠️  Błąd najlepszego: {e}")

        if is_regular:
            epoch_path = os.path.join(config.model_dir, f"model_latest.pt")
            try:
                torch.save(checkpoint_data, epoch_path)
                if os.path.exists(epoch_path):
                    size = os.path.getsize(epoch_path)
                    save_messages.append(f"📝 Latest: model_latest.pt ({size / 1024 / 1024:.1f} MB)")
                files_saved.append(epoch_path)
            except Exception as e:
                save_messages.append(f"⚠️  Błąd latest: {e}")

        if save_messages:
            print(f"\n{'─' * 50}")
            print("📦 PODSUMOWANIE ZAPISU:")
            for msg in save_messages:
                print(f"  {msg}")
            print(f"{'─' * 50}")

        return files_saved

    except Exception as e:
        print(f"\n❌ KRYTYCZNY BŁĄD w save_checkpoint: {e}")
        return []


# -------------------- Ładowanie danych --------------------
def load_and_preprocess_data(folder, books_folder=None):
    print(f"\n{'=' * 60}")
    print("📂 ŁADOWANIE DANYCH")
    print(f"{'=' * 60}")

    all_data = []
    data_files = glob.glob(os.path.join(folder, "*.json"))
    print(f"Znaleziono {len(data_files)} plików JSON")

    files_loaded = 0
    for i, file in enumerate(data_files):
        print_progress_bar(i, len(data_files), prefix="Ładowanie JSON:",
                           suffix=os.path.basename(file), length=30)
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    input_text = item.get("input", "").strip()
                    output_text = item.get("output", "").strip()
                    if 10 < len(input_text) < 300 and 10 < len(output_text) < 300:
                        all_data.append({"input": input_text, "output": output_text})
            files_loaded += 1
        except Exception as e:
            print(f"\n⚠️  Błąd przy wczytywaniu {file}: {e}")

    print_progress_bar(len(data_files), len(data_files), prefix="Ładowanie JSON:",
                       suffix=f"Zakończono ({files_loaded}/{len(data_files)})", length=30)

    if books_folder:
        book_files = glob.glob(os.path.join(books_folder, "*.txt"))
        print(f"\nZnaleziono {len(book_files)} plików tekstowych (książki)")

        for i, file in enumerate(book_files):
            print_progress_bar(i, len(book_files), prefix="Ładowanie książek:",
                               suffix=os.path.basename(file), length=30)
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                    content = TextPreprocessor.clean_text(content)

                    chunks = TextPreprocessor.split_into_chunks(content, chunk_size=config.context_size)

                    for j in range(len(chunks) - 1):
                        if len(chunks[j]) > 20 and len(chunks[j + 1]) > 20:
                            all_data.append({
                                "input": f"Kontynuuj tekst: {chunks[j][:100]}...",
                                "output": chunks[j + 1]
                            })

                print(f"\n  📖 {os.path.basename(file)}: {len(chunks)} fragmentów")

            except Exception as e:
                print(f"\n⚠️  Błąd przy wczytywaniu książki {file}: {e}")

        print_progress_bar(len(book_files), len(book_files), prefix="Ładowanie książek:",
                           suffix="Zakończono", length=30)

    print(f"\n✅ Załadowano {len(all_data)} próbek dialogów")
    return all_data


# -------------------- Słownik --------------------
def build_vocabulary(data, min_freq=2):
    print(f"\n{'=' * 60}")
    print("🔤 BUDOWANIE SŁOWNIKA")
    print(f"{'=' * 60}")

    all_text = ""
    print("Łączenie tekstów...")
    for i, item in enumerate(data):
        print_progress_bar(i, len(data), prefix="Przetwarzanie dialogów:",
                           suffix=f"{i}/{len(data)}", length=30)
        all_text += f"{item['input']}\n{item['output']}\n"

    print_progress_bar(len(data), len(data), prefix="Przetwarzanie dialogów:",
                       suffix="Zakończono", length=30)

    print("Liczenie częstotliwości...")
    char_counts = {}
    total_chars = len(all_text)
    for i, char in enumerate(all_text):
        if i % 100000 == 0:
            print_progress_bar(i, total_chars, prefix="Analiza znaków:",
                               suffix=f"{i / total_chars * 100:.1f}%", length=30)
        char_counts[char] = char_counts.get(char, 0) + 1

    print_progress_bar(total_chars, total_chars, prefix="Analiza znaków:",
                       suffix="Zakończono", length=30)

    SPECIAL_TOKENS = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
    common_chars = [c for c, count in char_counts.items()
                    if count >= min_freq or c in ' .,!?\nąćęłńóśźżĄĆĘŁŃÓŚŹŻ']

    chars = SPECIAL_TOKENS + sorted(common_chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    print(f"\n✅ Słownik: {len(chars)} znaków")
    print(f"📊 Rozkład:")
    print(f"  - Znaki specjalne: {len(SPECIAL_TOKENS)}")
    print(f"  - Znaki wspólne: {len(common_chars)}")
    print(f"  - Wszystkie unikalne: {len(char_counts)}")
    print(f"  - Całkowita liczba znaków w danych: {total_chars:,}")

    return stoi, itos, len(chars), total_chars


# -------------------- Model --------------------
class AdvancedChatModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = self.embedding(x)

        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)

        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        output = self.fc(x)
        return output, hidden


# -------------------- Trening --------------------
def train_model(config: ChatbotConfig):
    print(f"\n{'=' * 80}")
    print("🚀 ROZPOCZĘCIE TRENINGU")
    print(f"{'=' * 80}")
    print(f"📊 Konfiguracja:")
    print(f"  • Epoki: {config.epochs}")
    print(f"  • Batch: {config.batch}")
    print(f"  • Książki: {config.books}")
    print(f"  • Backup co: {config.backup_freq} epok")
    print(f"  • Główny model co: {config.main_save_freq} epok")
    print(f"{'=' * 80}")

    data = load_and_preprocess_data(config.data, config.books)

    stoi, itos, vocab_size, total_chars = build_vocabulary(data)

    # Obliczanie szacowanego czasu treningu
    print(f"\n{'=' * 60}")
    print("⏱️  SZACOWANIE CZASU TRENINGU")
    print(f"{'=' * 60}")

    # Całkowita liczba tokenów do przetworzenia
    total_tokens = total_chars * config.epochs
    print(f"📊 Statystyki tokenów:")
    print(f"  • Tokenów w danych: {total_chars:,}")
    print(f"  • Epoki: {config.epochs}")
    print(f"  • Całkowite tokeny do przetworzenia: {total_tokens:,}")

    estimated_seconds, time_str = estimate_training_time(config, total_chars)
    print(f"  • Szacowany czas treningu: {time_str}")
    print(f"  • Szacowana wydajność: {config.estimated_tokens_per_second:,} tokenów/sekundę")

    if estimated_seconds > 3600:  # Jeśli więcej niż godzina
        print(f"\n⚠️  UWAGA: Szacowany czas treningu przekracza 1 godzinę!")
        print(f"   Rozważ zmniejszenie liczby epok lub rozmiaru danych.")

    confirm = input(f"\n🔸 Czy kontynuować trening? (tak/nie): ").strip().lower()
    if confirm not in ['t', 'tak', 'y', 'yes']:
        print("❌ Trening anulowany.")
        return

    print(f"\n{'=' * 60}")
    print("🔧 PRZYGOTOWYWANIE DANYCH TRENINGOWYCH")
    print(f"{'=' * 60}")

    def prepare_samples():
        samples = []
        total_items = len(data)

        for idx, item in enumerate(data):
            if idx % 100 == 0:
                print_progress_bar(idx, total_items, prefix="Tokenizacja:",
                                   suffix=f"{idx}/{total_items}", length=30)

            text = f"<BOS>{item['input']}\n{item['output']}<EOS>"
            tokens = [stoi.get(c, stoi['<UNK>']) for c in text]

            if len(tokens) > config.max_length:
                tokens = tokens[:config.max_length]
                tokens[-1] = stoi['<EOS>']

            padded = tokens + [stoi['<PAD>']] * (config.max_length - len(tokens))

            x = torch.tensor(padded[:-1], dtype=torch.long)
            y = torch.tensor(padded[1:], dtype=torch.long)
            samples.append((x, y))

        print_progress_bar(total_items, total_items, prefix="Tokenizacja:",
                           suffix="Zakończono", length=30)

        print(f"\n✅ Przygotowano {len(samples)} próbek")
        print(f"📏 Długość sekwencji: {config.max_length} tokenów")
        return samples

    dataset = prepare_samples()
    random.shuffle(dataset)

    split_idx = int(0.85 * len(dataset))
    train_data, val_data = dataset[:split_idx], dataset[split_idx:]

    print(f"\n📊 Podział danych:")
    print(f"  • Train: {len(train_data)} próbek ({len(train_data) / len(dataset) * 100:.1f}%)")
    print(f"  • Val:   {len(val_data)} próbek ({len(val_data) / len(dataset) * 100:.1f}%)")
    print(f"  • Total: {len(dataset)} próbek")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Używane urządzenie: {device}")
    if device.type == 'cuda':
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • Pamięć: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    print(f"\n🧠 Inicjalizacja modelu...")
    model = AdvancedChatModel(
        vocab_size=vocab_size,
        hidden_size=config.hidden,
        num_layers=config.layers,
        dropout=config.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model utworzony:")
    print(f"  • Parametry: {total_params:,} (trenowalne: {trainable_params:,})")
    print(f"  • Rozmiar: ~{total_params * 4 / 1024 ** 2:.1f} MB (float32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.lr * 0.01
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=stoi['<PAD>'])

    best_val_loss = float('inf')
    training_start_time = time.time()
    tokens_processed = 0
    start_time = time.time()

    print(f"\n{'=' * 80}")
    print("🔥 ROZPOCZĘCIE TRENINGU")
    print(f"{'=' * 80}")

    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        batches_processed = 0
        epoch_tokens = 0

        random.shuffle(train_data)

        print(f"\n📈 Epoka {epoch + 1}/{config.epochs}")
        print(f"{'─' * 40}")

        for i in range(0, len(train_data), config.batch):
            batch_idx = i // config.batch + 1
            total_batches = len(train_data) // config.batch + 1

            batch_samples = train_data[i:i + config.batch]
            if not batch_samples:
                continue

            x_batch = torch.stack([x for x, y in batch_samples]).to(device)
            y_batch = torch.stack([y for x, y in batch_samples]).to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)

            loss = loss_fn(
                logits.view(-1, vocab_size),
                y_batch.view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            batches_processed += 1

            # Aktualizuj liczbę przetworzonych tokenów
            batch_tokens = x_batch.numel()
            epoch_tokens += batch_tokens
            tokens_processed += batch_tokens

            if batch_idx % 10 == 0 or batch_idx == total_batches:
                current_time = time.time()
                elapsed_time = current_time - start_time
                tokens_per_second = tokens_processed / elapsed_time if elapsed_time > 0 else 0

                # Oblicz pozostały czas
                tokens_remaining = total_chars * config.epochs - tokens_processed
                if tokens_per_second > 0:
                    time_remaining = tokens_remaining / tokens_per_second
                    if time_remaining < 60:
                        remaining_str = f"{time_remaining:.1f}s"
                    elif time_remaining < 3600:
                        remaining_str = f"{time_remaining / 60:.1f}m"
                    else:
                        remaining_str = f"{time_remaining / 3600:.1f}h"
                else:
                    remaining_str = "obliczanie..."

                print_progress_bar(batch_idx, total_batches,
                                   prefix=f"Batch {batch_idx}/{total_batches}:",
                                   suffix=f"loss: {loss.item():.4f} | tok/s: {tokens_per_second:.0f} | pozostało: {remaining_str}",
                                   length=30)

        avg_train_loss = total_train_loss / max(batches_processed, 1)

        model.eval()
        total_val_loss = 0
        val_batches = 0

        print(f"\n{'─' * 40}")
        print("🧪 WALIDACJA")

        with torch.no_grad():
            for i in range(0, len(val_data), config.batch):
                batch_samples = val_data[i:i + config.batch]
                if not batch_samples:
                    continue

                x_batch = torch.stack([x for x, y in batch_samples]).to(device)
                y_batch = torch.stack([y for x, y in batch_samples]).to(device)

                logits, _ = model(x_batch)
                loss = loss_fn(
                    logits.view(-1, vocab_size),
                    y_batch.view(-1)
                )
                total_val_loss += loss.item()
                val_batches += 1

                batch_idx = i // config.batch + 1
                total_val_batches = len(val_data) // config.batch + 1
                if batch_idx % 5 == 0 or batch_idx == total_val_batches:
                    print_progress_bar(batch_idx, total_val_batches,
                                       prefix="Walidacja:",
                                       suffix=f"loss: {loss.item():.4f}",
                                       length=30)

        avg_val_loss = total_val_loss / max(val_batches, 1)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time

        # Oblicz rzeczywistą wydajność
        actual_tokens_per_second = epoch_tokens / epoch_time if epoch_time > 0 else 0

        # Aktualizuj szacowaną wydajność w konfiguracji
        if actual_tokens_per_second > 0:
            config.estimated_tokens_per_second = actual_tokens_per_second

        print(f"\n{'─' * 40}")
        print("📊 STATYSTYKI EPOKI")
        print(f"{'─' * 40}")
        print(f"Epoka:          {epoch + 1:3d}/{config.epochs}")
        print(f"Train Loss:     {avg_train_loss:.6f}")
        print(f"Val Loss:       {avg_val_loss:.6f}")
        print(f"Learning Rate:  {current_lr:.6f}")
        print(f"Tokenów w epoce: {epoch_tokens:,}")
        print(f"Czas epoki:     {epoch_time:.1f}s")
        print(f"Tokenów/sekundę: {actual_tokens_per_second:.0f}")
        print(f"Czas całkowity: {total_time / 60:.1f}m")

        # Oblicz pozostały czas
        epochs_remaining = config.epochs - epoch - 1
        if actual_tokens_per_second > 0:
            time_remaining = (total_chars * epochs_remaining) / actual_tokens_per_second
            if time_remaining < 60:
                remaining_str = f"{time_remaining:.1f} sekund"
            elif time_remaining < 3600:
                remaining_str = f"{time_remaining / 60:.1f} minut"
            else:
                remaining_str = f"{time_remaining / 3600:.1f} godzin"
            print(f"Pozostało:      {remaining_str}")

        print_progress_bar(epoch + 1, config.epochs,
                           prefix="Całkowity postęp:",
                           suffix=f"Epoka {epoch + 1}/{config.epochs}",
                           length=40)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'stoi': stoi,
            'itos': itos,
            'vocab_size': vocab_size,
            'hidden_size': config.hidden,
            'num_layers': config.layers,
            'config': vars(config),
            'timestamp': datetime.datetime.now().isoformat(),
            'tokens_processed': tokens_processed,
            'actual_tokens_per_second': actual_tokens_per_second
        }

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f"\n🏆 NOWY NAJLEPSZY WYNIK! Val loss: {avg_val_loss:.6f}")

        saved_files = save_checkpoint(
            config=config,
            checkpoint_data=checkpoint,
            epoch=epoch + 1,
            val_loss=avg_val_loss,
            is_best=is_best,
            is_regular=True
        )

        print(f"\n{'=' * 80}")

    training_total_time = time.time() - training_start_time

    # Oblicz rzeczywistą wydajność
    actual_tokens_per_second_total = tokens_processed / training_total_time if training_total_time > 0 else 0

    print(f"\n{'=' * 80}")
    print("🎉 TRENING ZAKOŃCZONY!")
    print(f"{'=' * 80}")
    print(f"📊 PODSUMOWANIE:")
    print(f"  • Czas całkowity: {training_total_time / 60:.1f} minut")
    print(f"  • Najlepszy val loss: {best_val_loss:.6f}")
    print(f"  • Średni czas na epokę: {training_total_time / config.epochs:.1f}s")
    print(f"  • Przetworzone tokeny: {tokens_processed:,}")
    print(f"  • Rzeczywista wydajność: {actual_tokens_per_second_total:.0f} tokenów/sekundę")
    print(f"\n💾 ZAPISANE PLIKI:")
    print(f"  • Model główny:     {config.model_path}")
    print(f"  • Model najlepszy:  {os.path.join(config.model_dir, 'model_best.pt')}")
    print(f"  • Model ostatni:    {os.path.join(config.model_dir, 'model_latest.pt')}")
    print(f"  • Backupy:          {config.backup_dir}/")
    print(f"  • Liczba backupów:  {len(glob.glob(os.path.join(config.backup_dir, '*.pt')))}")
    print(f"{'=' * 80}")


# -------------------- Generowanie --------------------
def generate_response(model, prompt, stoi, itos, max_length=100, temperature=0.7):
    model.eval()
    device = next(model.parameters()).device

    if not prompt.startswith('<BOS>'):
        prompt = f'<BOS>{prompt}'

    tokens = [stoi.get(c, stoi.get('<UNK>', 0)) for c in prompt]
    x = torch.tensor(tokens).unsqueeze(0).to(device)

    generated = []
    hidden = None

    print("🤖 Generowanie odpowiedzi...")
    with torch.no_grad():
        for i in range(max_length):
            print_progress_bar(i, max_length, prefix="Generowanie:",
                               suffix=f"token {i}/{max_length}", length=30)

            logits, hidden = model(x, hidden)
            logits = logits[0, -1] / temperature

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == stoi.get('<EOS>', -1):
                print_progress_bar(max_length, max_length, prefix="Generowanie:",
                                   suffix="Zakończono (EOS)", length=30)
                break

            generated.append(next_token)
            x = torch.tensor([[next_token]]).to(device)

            if i == max_length - 1:
                print_progress_bar(max_length, max_length, prefix="Generowanie:",
                                   suffix="Zakończono (max)", length=30)

    response = ''.join(itos.get(idx, '?') for idx in generated)
    return response


# -------------------- Tryb uczenia przez rozmowę --------------------
def train_talk_mode(config: ChatbotConfig):
    print(f"\n{'=' * 60}")
    print("🎓 TRYB UCZENIA PRZEZ ROZMOWĘ")
    print(f"{'=' * 60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(config.model_path):
        print(f"📥 Ładowanie istniejącego modelu...")
        checkpoint = torch.load(config.model_path, map_location='cpu', weights_only=False)

        model = AdvancedChatModel(
            vocab_size=checkpoint['vocab_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        stoi = checkpoint['stoi']
        itos = checkpoint['itos']
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr * 0.1)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"✅ Wczytano model z epoki {checkpoint.get('epoch', 'N/A')}")
    else:
        print("❌ Nie znaleziono istniejącego modelu. Rozpocznij najpierw normalny trening.")
        return

    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=stoi['<PAD>'])

    conversation_history = []
    learning_examples = []

    print(f"\n💬 Rozpocznij rozmowę! Model będzie się uczył na podstawie Twoich poprawek.")
    print(f"📝 Wpisz '--train' po odpowiedzi aby dodać ją do treningu")

    while True:
        try:
            user_input = input("\n👤 Ty: ").strip()

            if user_input.lower() == 'quit':
                print(f"\n{'=' * 60}")
                print("👋 Zakończono tryb uczenia")
                print(f"{'=' * 60}")
                break
            elif user_input.lower() == 'skip':
                print("⏭️  Pominięto")
                continue
            elif user_input.lower() == 'save':
                checkpoint = {
                    'epoch': checkpoint.get('epoch', 0) + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': 0.0,
                    'train_loss': 0.0,
                    'stoi': stoi,
                    'itos': itos,
                    'vocab_size': len(stoi),
                    'hidden_size': config.hidden,
                    'num_layers': config.layers,
                    'config': vars(config),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'learning_examples': len(learning_examples)
                }

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(config.model_dir, f"model_traintalk_{timestamp}.pt")
                torch.save(checkpoint, save_path)
                print(f"💾 Model zapisany jako: {save_path}")
                continue
            elif user_input.lower() == 'info':
                print(f"\n{'─' * 50}")
                print("📊 INFORMACJE O MODELU:")
                print(f"{'─' * 50}")
                print(f"Przykłady treningowe: {len(learning_examples)}")
                print(f"Rozmiar słownika: {len(stoi)}")
                print(f"Rozmiar historii: {len(conversation_history)}")
                print(f"{'─' * 50}")
                continue
            elif not user_input:
                continue

            print(f"\n{'─' * 30}")
            print("🤖 Bot myśli...")
            response = generate_response(model, user_input, stoi, itos, max_length=100, temperature=0.7)
            print(f"\n🤖 Bot: {response}")
            print(f"{'─' * 30}")

            correction = input(
                "\n✏️  Popraw odpowiedź (lub Enter aby zaakceptować, '--train' aby dodać do treningu): ").strip()

            if correction == "":
                print("✅ Zaakceptowano odpowiedź bota")
                conversation_history.append({"input": user_input, "output": response})
            elif correction.lower() == '--train':
                learning_examples.append({"input": user_input, "output": response})
                conversation_history.append({"input": user_input, "output": response})
                print(f"✅ Dodano do przykładów treningowych. Razem: {len(learning_examples)}")

                if len(learning_examples) >= 5:
                    print(f"\n🎯 Wykonuję trening na {len(learning_examples)} przykładach...")
                    train_on_examples(model, optimizer, loss_fn, learning_examples, stoi, device, config)
                    learning_examples.clear()
                    print("✅ Trening zakończony")

            elif correction:
                conversation_history.append({"input": user_input, "output": correction})
                learning_examples.append({"input": user_input, "output": correction})
                print(f"✅ Użyto poprawionej odpowiedzi. Przykłady treningowe: {len(learning_examples)}")

                print(f"\n🎯 Uczenie na poprawionej odpowiedzi...")
                train_single_example(model, optimizer, loss_fn, user_input, correction, stoi, device, config)
                print("✅ Nauczono na podstawie korekty")

        except KeyboardInterrupt:
            print(f"\n{'=' * 60}")
            print("🛑 Zakończono tryb uczenia")
            print(f"{'=' * 60}")
            break
        except Exception as e:
            print(f"\n❌ Błąd: {e}")


def train_single_example(model, optimizer, loss_fn, input_text, output_text, stoi, device, config):
    model.train()

    text = f"<BOS>{input_text}\n{output_text}<EOS>"
    tokens = [stoi.get(c, stoi['<UNK>']) for c in text]

    if len(tokens) > config.max_length:
        tokens = tokens[:config.max_length]
        tokens[-1] = stoi['<EOS>']

    padded = tokens + [stoi['<PAD>']] * (config.max_length - len(tokens))

    x = torch.tensor(padded[:-1], dtype=torch.long).unsqueeze(0).to(device)
    y = torch.tensor(padded[1:], dtype=torch.long).unsqueeze(0).to(device)

    optimizer.zero_grad()
    logits, _ = model(x)

    loss = loss_fn(
        logits.view(-1, len(stoi)),
        y.view(-1)
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def train_on_examples(model, optimizer, loss_fn, examples, stoi, device, config):
    model.train()

    total_loss = 0
    batch_size = min(4, len(examples))

    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i:i + batch_size]

        x_batch = []
        y_batch = []

        for example in batch_examples:
            text = f"<BOS>{example['input']}\n{example['output']}<EOS>"
            tokens = [stoi.get(c, stoi['<UNK>']) for c in text]

            if len(tokens) > config.max_length:
                tokens = tokens[:config.max_length]
                tokens[-1] = stoi['<EOS>']

            padded = tokens + [stoi['<PAD>']] * (config.max_length - len(tokens))

            x_batch.append(padded[:-1])
            y_batch.append(padded[1:])

        x = torch.tensor(x_batch, dtype=torch.long).to(device)
        y = torch.tensor(y_batch, dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits, _ = model(x)

        loss = loss_fn(
            logits.view(-1, len(stoi)),
            y.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(examples) / batch_size, 1)


# -------------------- Chat --------------------
def chat_mode(config: ChatbotConfig):
    print(f"\n{'=' * 60}")
    print("🤖 TRYB ROZMOWY")
    print(f"{'=' * 60}")

    if not os.path.exists(config.model_dir):
        print("❌ Nie znaleziono folderu z modelami!")
        return

    print("\n📂 Dostępne modele:")
    print(f"{'─' * 50}")

    models = []
    possible_models = [
        ("model.pt (główny)", config.model_path),
        ("model_best.pt (najlepszy)", os.path.join(config.model_dir, "model_best.pt")),
        ("model_latest.pt (ostatni)", os.path.join(config.model_dir, "model_latest.pt"))
    ]

    traintalk_models = glob.glob(os.path.join(config.model_dir, "model_traintalk_*.pt"))
    for model_file in sorted(traintalk_models, reverse=True)[:5]:
        possible_models.append((f"traintalk: {os.path.basename(model_file)}", model_file))

    for name, path in possible_models:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                epoch = checkpoint.get('epoch', 'N/A')
                val_loss = checkpoint.get('val_loss', 'N/A')
                timestamp = checkpoint.get('timestamp', 'N/A')
                if isinstance(timestamp, str):
                    timestamp = timestamp.split('T')[0]
                models.append((f"{len(models) + 1}. {name}", path, epoch, val_loss, timestamp))
            except:
                models.append((f"{len(models) + 1}. {name} (uszkodzony)", path, 'N/A', 'N/A', 'N/A'))

    backup_files = glob.glob(os.path.join(config.backup_dir, "*.pt"))
    for i, backup in enumerate(sorted(backup_files, reverse=True)[:10], start=len(models) + 1):
        filename = os.path.basename(backup)
        try:
            checkpoint = torch.load(backup, map_location='cpu', weights_only=False)
            epoch = checkpoint.get('epoch', 'N/A')
            val_loss = checkpoint.get('val_loss', 'N/A')
            timestamp = checkpoint.get('timestamp', 'N/A')
            if isinstance(timestamp, str):
                timestamp = timestamp.split('T')[0]
            models.append((f"{i}. {filename}", backup, epoch, val_loss, timestamp))
        except:
            models.append((f"{i}. {filename} (uszkodzony)", backup, 'N/A', 'N/A', 'N/A'))

    if not models:
        print("❌ Nie znaleziono żadnych modeli!")
        return

    for name, _, epoch, val_loss, timestamp in models:
        if val_loss != 'N/A':
            print(f"  {name:40} | epoka: {epoch:4} | loss: {val_loss:.4f} | data: {timestamp}")
        else:
            print(f"  {name:40} | epoka: {epoch:4} | loss: {val_loss} | data: {timestamp}")

    print(f"{'─' * 50}")

    try:
        choice = input("\n🎯 Wybierz model (nr) lub Enter dla domyślnego: ").strip()
        if choice == "":
            model_path = config.model_path
            print(f"🔹 Wybrano domyślny: {os.path.basename(model_path)}")
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model_path = models[idx][1]
                print(f"🔹 Wybrano: {models[idx][0].split('. ')[1]}")
            else:
                print("⚠️  Nieprawidłowy wybór, używam domyślnego")
                model_path = config.model_path
    except:
        model_path = config.model_path

    if not os.path.exists(model_path):
        print(f"❌ Nie znaleziono modelu: {model_path}")
        return

    try:
        print(f"\n{'─' * 50}")
        print("📥 Ładowanie modelu...")
        print_progress_bar(0, 3, prefix="Wczytywanie:", suffix="inicjalizacja", length=30)
        time.sleep(0.2)

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print_progress_bar(1, 3, prefix="Wczytywanie:", suffix="checkpoint", length=30)
        time.sleep(0.2)

        model = AdvancedChatModel(
            vocab_size=checkpoint['vocab_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print_progress_bar(2, 3, prefix="Wczytywanie:", suffix="model", length=30)
        time.sleep(0.2)

        stoi = checkpoint['stoi']
        itos = checkpoint['itos']

        print_progress_bar(3, 3, prefix="Wczytywanie:", suffix="GOTOWE!", length=30)

        print(f"\n{'─' * 50}")
        print("✅ Model wczytany pomyślnie!")
        print(f"{'─' * 50}")
        print(f"📄 Plik:        {os.path.basename(model_path)}")
        print(f"🎯 Epoka:       {checkpoint.get('epoch', 'N/A')}")
        print(f"📉 Val loss:    {checkpoint.get('val_loss', 'N/A'):.6f}")
        print(f"📈 Train loss:  {checkpoint.get('train_loss', 'N/A'):.6f}")
        print(f"🔤 Słownik:     {len(stoi)} tokenów")
        print(f"📅 Data:        {checkpoint.get('timestamp', 'N/A')}")
        print(f"🧠 Warstwy:     {checkpoint.get('num_layers', 'N/A')}")
        print(f"⚙️  Neurony:     {checkpoint.get('hidden_size', 'N/A')}")
        if 'tokens_processed' in checkpoint:
            print(f"🔢 Tokeny:       {checkpoint.get('tokens_processed', 'N/A'):,}")
        if 'actual_tokens_per_second' in checkpoint:
            print(f"⚡ Wydajność:    {checkpoint.get('actual_tokens_per_second', 'N/A'):.0f} tok/s")
        print(f"{'─' * 50}")
        print("💬 Rozpocznij rozmowę! (wpisz 'quit' aby wyjść)")
        print(f"{'=' * 60}")

        conversation_context = []
        while True:
            try:
                user_input = input("\n👤 Ty: ").strip()

                if user_input.lower() == 'quit':
                    print(f"\n{'=' * 60}")
                    print("👋 Do widzenia!")
                    print(f"{'=' * 60}")
                    break
                elif user_input.lower() == 'reset':
                    conversation_context.clear()
                    print("🔄 Kontekst zresetowany")
                    continue
                elif user_input.lower() == 'list':
                    print("\n")
                    chat_mode(config)
                    break
                elif user_input.lower() == 'info':
                    print(f"\n{'─' * 50}")
                    print("📊 INFORMACJE O MODELU:")
                    print(f"{'─' * 50}")
                    print(f"Plik: {os.path.basename(model_path)}")
                    print(f"Epoka: {checkpoint.get('epoch', 'N/A')}")
                    print(f"Val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
                    print(f"Data: {checkpoint.get('timestamp', 'N/A')}")
                    print(f"Rozmiar kontekstu: {len(conversation_context)}")
                    print(f"{'─' * 50}")
                    continue
                elif not user_input:
                    continue

                if conversation_context:
                    context = "\n".join(conversation_context[-3:])
                    full_input = f"{context}\n{user_input}"
                else:
                    full_input = user_input

                print(f"\n{'─' * 30}")
                response = generate_response(
                    model,
                    full_input,
                    stoi,
                    itos,
                    max_length=150,
                    temperature=0.7
                )
                print(f"\n{'─' * 30}")
                print(f"🤖 Bot: {response}")
                print(f"{'─' * 30}")

                conversation_context.append(f"Ty: {user_input}")
                conversation_context.append(f"Bot: {response}")

                if len(conversation_context) > 20:
                    conversation_context = conversation_context[-20:]

            except KeyboardInterrupt:
                print(f"\n{'=' * 60}")
                print("🛑 Zakończono chat")
                print(f"{'=' * 60}")
                break
            except Exception as e:
                print(f"\n❌ Błąd podczas generowania: {e}")

    except Exception as e:
        print(f"❌ Błąd przy wczytywaniu modelu: {e}")


# -------------------- Główna logika --------------------
if __name__ == "__main__":
    print(f"{'=' * 60}")
    print("🧠 CHATBOT LSTM - SYSTEM TRENINGU I ROZMOWY")
    print(f"{'=' * 60}")
    print(f"PyTorch wersja: {torch.__version__}")
    print(f"CUDA dostępne:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 60}")

    config = ChatbotConfig(
        learn=False,
        talk=False,
        traintalk=True
    )

    if len(sys.argv) > 1:
        if '--traintalk' in sys.argv:
            config.traintalk = True
            config.learn = False
            config.talk = False
        if '--train' in sys.argv:
            config.learn = True
            config.traintalk = False
            config.talk = False
        if '--talk' in sys.argv:
            config.talk = True
            config.learn = False
            config.traintalk = False

    if os.path.exists(config.config_path):
        if config.load_config():
            print("✅ Wczytano konfigurację z pliku")
    else:
        print("ℹ️  Brak zapisanej konfiguracji, używam domyślnej")

    print(f"\n{'=' * 60}")
    print("⚙️  KONFIGURACJA")
    print(f"{'=' * 60}")
    print(f"Tryb:           {'TRENING' if config.learn else 'ROZMOWA' if config.talk else 'UCZENIE PRZEZ ROZMOWĘ'}")
    print(f"Epoki:          {config.epochs}")
    print(f"Batch:          {config.batch}")
    print(f"Neurony:        {config.hidden}")
    print(f"Warstwy LSTM:   {config.layers}")
    print(f"Learning rate:  {config.lr}")
    print(f"Max długość:    {config.max_length}")
    print(f"Backup co:      {config.backup_freq} epok")
    print(f"Główny model co:{config.main_save_freq} epok")
    print(f"Folder danych:  {config.data}")
    print(f"Folder książek: {config.books}")
    print(f"Folder modeli:  {config.model_dir}")
    print(f"Folder backup:  {config.backup_dir}")
    print(f"Szac. wydajność:{config.estimated_tokens_per_second:,} tok/s")
    print(f"{'=' * 60}")

    if config.learn:
        train_model(config)
    elif config.talk:
        chat_mode(config)
    elif config.traintalk:
        train_talk_mode(config)
    else:
        print("❌ Ustaw w konfiguracji learn=True, talk=True lub traintalk=True")
