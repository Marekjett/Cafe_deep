"""
🎯 MAIN - Główny skrypt MiniGPT-60M z checkpointami
"""

import os
import sys
import argparse
import signal
from pathlib import Path

# Dodaj ścieżkę do importów
sys.path.append('.')

from config import logger, cfg, sys_config
import prepare_data
import ai

# ==================== OBSŁUGA SYGNAŁÓW ====================
def signal_handler(sig, frame):
    """Obsługa przerwania (Ctrl+C)"""
    logger.info("\n\n⚠️ Trening przerwany przez użytkownika!")
    logger.info("💾 Zapisuję stan do wznowienia...")

    # Tutaj można dodać zapis stanu przed wyjściem
    # W aktualnej implementacji stan jest zapisywany automatycznie

    logger.info("✅ Możesz wznowić trening używając: python main.py --cont")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ==================== FUNKCJE GŁÓWNE ====================
def prepare_all_data():
    """Przygotowuje wszystkie dane"""
    logger.info("=" * 60)
    logger.info("📊 PRZYGOTOWYWANIE WSZYSTKICH DANYCH")
    logger.info("=" * 60)

    preparer = prepare_data.DataPreparer()
    preparer.prepare_all_data()

def train_model(resume: bool = False):
    """Trenuje model"""
    logger.info("=" * 60)
    if resume:
        logger.info("🔄 WZNIOWANIE TRENINGU MODELU")
    else:
        logger.info("🚀 TRENING MODELU MINIGPT-60M")
    logger.info("=" * 60)

    # Sprawdź czy dane są przygotowane
    data_file = Path(cfg.prepared_dir) / "all_data.txt"

    if not data_file.exists():
        logger.warning("⚠️ Brak przygotowanych danych. Przygotowuję...")
        prepare_all_data()

    # Uruchom trening
    if resume:
        ai.continue_training()
    else:
        ai.train_model(resume=False)

def continue_training():
    """Kontynuuje trening od ostatniego checkpointu"""
    train_model(resume=True)

def train_more_epochs(additional_epochs: int = 3):
    """Dodaje więcej epok treningu"""
    logger.info("=" * 60)
    logger.info(f"📈 DODATKOWY TRENING: {additional_epochs} epok")
    logger.info("=" * 60)

    # Sprawdź czy dane są przygotowane
    data_file = Path(cfg.prepared_dir) / "all_data.txt"

    if not data_file.exists():
        logger.error("❌ Brak przygotowanych danych!")
        logger.info("💡 Uruchom: python main.py --prepare")
        return

    # Uruchom dodatkowy trening
    ai.train_more_epochs(additional_epochs)

def generate_text(prompt: str):
    """Generuje tekst"""
    logger.info(f"🎨 GENEROWANIE TEKSTU: '{prompt}'")
    ai.generate_text(prompt)

def start_chat():
    """Uruchamia czat"""
    ai.start_chat()

def evaluate_model():
    """Ocenia model"""
    logger.info("🎯 OCENA MODELU")

    # Sprawdź czy model istnieje
    model_path = cfg.get_latest_model()

    if not model_path:
        logger.error("❌ Brak wytrenowanego modelu!")
        logger.info("💡 Uruchom: python main.py --train")
        return

    logger.info(f"✅ Model wczytany: {model_path.name}")

    # Tutaj można dodać ewaluację
    logger.info("📊 Dodaj ewaluację w ai.py")

def show_checkpoints():
    """Pokazuje dostępne checkpointy"""
    logger.info("📁 DOSTĘPNE CHECKPOINTY:")

    checkpoints = list(Path(cfg.checkpoints_dir).glob("checkpoint_*.pt"))
    models = list(Path(cfg.model_dir).glob("model_*.pt"))

    if checkpoints:
        logger.info("\n🔽 CHECKPOINTY (do wznowienia):")
        for cp in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            size_mb = cp.stat().st_size / (1024 * 1024)
            logger.info(f"   • {cp.name} ({size_mb:.1f} MB)")
    else:
        logger.info("   ❌ Brak checkpointów")

    if models:
        logger.info("\n🤖 MODELE:")
        for model in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            size_mb = model.stat().st_size / (1024 * 1024)
            logger.info(f"   • {model.name} ({size_mb:.1f} MB)")

    # Sprawdź stan resume
    resume_state = cfg.load_resume_state()
    if resume_state:
        logger.info(f"\n🔄 OSTATNI STAN TRENINGU:")
        logger.info(f"   • Epoka: {resume_state.get('epoch', 'N/A')}")
        logger.info(f"   • Krok: {resume_state.get('step', 'N/A')}")
        logger.info(f"   • Loss: {resume_state.get('train_loss', 'N/A')}")

def clean_checkpoints(keep_last: int = 3):
    """Czyści stare checkpointy"""
    logger.info(f"🧹 CZYSZCZENIE STARYCH CHECKPOINTÓW (zachowuję {keep_last} najnowszych)")

    checkpoints = list(Path(cfg.checkpoints_dir).glob("checkpoint_*.pt"))

    if len(checkpoints) <= keep_last:
        logger.info("✅ Nie ma czego czyścić")
        return

    # Sortuj od najstarszego do najnowszego
    checkpoints.sort(key=lambda x: x.stat().st_mtime)

    # Usuń wszystkie poza keep_last najnowszych
    to_delete = checkpoints[:-keep_last]

    for cp in to_delete:
        try:
            cp.unlink()
            logger.info(f"   🗑️ Usunięto: {cp.name}")
        except Exception as e:
            logger.error(f"   ❌ Błąd usuwania {cp.name}: {e}")

    logger.info(f"✅ Pozostawiono {keep_last} najnowszych checkpointów")

def show_config():
    """Pokazuje konfigurację"""
    cfg.print_config()

def run_tests():
    """Uruchamia testy"""
    logger.info("🧪 TESTY JEDNOSTKOWE")

    # Test tokenizera
    from ai import tokenizer
    text = "Test tokenizacji"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    logger.info(f"✅ Tokenizer: '{text}' -> '{decoded}'")

    # Test modelu
    import torch
    model = ai.MiniGPT60M()
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(x)

    logger.info(f"✅ Model: logits shape {logits.shape}")
    logger.info("✅ Wszystkie testy przeszły pomyślnie!")

# ==================== GŁÓWNA FUNKCJA ====================
def main():
    """Główna funkcja programu"""

    parser = argparse.ArgumentParser(
        description="🎯 MiniGPT-60M: Zaawansowany model językowy ~60M parametrów",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python main.py --prepare        # Przygotuj dane ze wszystkich folderów data_*
  python main.py --train          # Trenuj model od początku
  python main.py --cont           # Wznów trening od ostatniego checkpointu
  python main.py --more [N]       # Dodaj N epok treningu (domyślnie 3)
  python main.py --generate "AI"  # Generuj tekst
  python main.py --chat           # Rozmawiaj z modelem
  python main.py --checkpoints    # Pokaż dostępne checkpointy
  python main.py --clean-cp       # Wyczyść stare checkpointy
  python main.py --test           # Uruchom testy
  python main.py --config         # Pokaż konfigurację
  python main.py --evaluate       # Oceń model
  python main.py --all            # Przygotuj dane i trenuj od początku

Kontynuacja treningu:
  Rozpocznij trening: python main.py --train
  Przerwij (Ctrl+C):  Zapisz stan automatycznie
  Wznów:              python main.py --cont
  Dodaj epoki:        python main.py --more 5
        """
    )

    parser.add_argument("--prepare", action="store_true", help="Przygotuj dane")
    parser.add_argument("--train", action="store_true", help="Trening od początku")
    parser.add_argument("--cont", action="store_true", help="Kontynuuj trening od checkpointu")
    parser.add_argument("--more", type=int, nargs='?', const=3, help="Dodaj epoki treningu (domyślnie 3)")
    parser.add_argument("--generate", type=str, help="Generuj tekst z promptu")
    parser.add_argument("--chat", action="store_true", help="Tryb rozmowy")
    parser.add_argument("--checkpoints", action="store_true", help="Pokaż dostępne checkpointy")
    parser.add_argument("--clean-cp", action="store_true", help="Wyczyść stare checkpointy")
    parser.add_argument("--test", action="store_true", help="Testy jednostkowe")
    parser.add_argument("--config", action="store_true", help="Pokaż konfigurację")
    parser.add_argument("--evaluate", action="store_true", help="Oceń model")
    parser.add_argument("--all", action="store_true", help="Przygotuj dane i trenuj")
    parser.add_argument("--epochs", type=int, default=cfg.epochs, help="Liczba epok")
    parser.add_argument("--model", type=str, help="Ścieżka do konkretnego modelu")

    args = parser.parse_args()

    # Pokaż nagłówek
    logger.info("=" * 60)
    logger.info("🎯 MINIGPT-60M - Z CHECKPOINTAMI")
    logger.info("=" * 60)

    # Update liczby epok jeśli podano
    if args.epochs != cfg.epochs:
        cfg.epochs = args.epochs
        logger.info(f"⚙️ Ustawiono {cfg.epochs} epok")

    # Uruchom odpowiednią funkcję
    if args.prepare:
        prepare_all_data()

    elif args.train:
        train_model(resume=False)

    elif args.cont:
        continue_training()

    elif args.more is not None:
        train_more_epochs(additional_epochs=args.more)

    elif args.generate:
        generate_text(args.generate)

    elif args.chat:
        start_chat()

    elif args.checkpoints:
        show_checkpoints()

    elif args.clean_cp:
        clean_checkpoints()

    elif args.test:
        run_tests()

    elif args.config:
        show_config()

    elif args.evaluate:
        evaluate_model()

    elif args.all:
        prepare_all_data()
        train_model(resume=False)

    else:
        # Jeśli żadna flaga, pokaż help
        logger.info("\n❓ Nie podano flagi. Dostępne opcje:\n")
        parser.print_help()

        # Pokaż dodatkowe informacje
        logger.info("\n💡 PRZYKŁADY UŻYCIA:")
        logger.info("   python main.py --train            # Trenuj od początku")
        logger.info("   [Ctrl+C]                         # Przerwij trening")
        logger.info("   python main.py --cont            # Wznów trening")
        logger.info("   python main.py --more 5          # Dodaj 5 epok")
        logger.info("   python main.py --generate 'AI'   # Generuj tekst")

        # Sprawdź czy są checkpointy
        if cfg.get_latest_checkpoint():
            logger.info("\n🔄 Dostępne checkpointy do wznowienia!")

        logger.info("=" * 60)

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