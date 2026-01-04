#!/usr/bin/env python3
"""
Napraw i uruchom chatbot
"""

import subprocess
import os


def main():
    print("🔧 NAPRAWA I URUCHOMIENIE CHATBOTA")

    # 1. Sprawdź foldery
    if not os.path.exists("model"):
        print("❌ Brak folderu 'model/'")
        print("Tworzę...")
        os.makedirs("model", exist_ok=True)

    if not os.path.exists("data"):
        print("❌ Brak folderu 'data/'")
        print("Tworzę...")
        os.makedirs("data", exist_ok=True)
        print("Dodaj pliki JSON do folderu 'data/'")

    # 2. Sprawdź czy jest model
    model_files = [
        "model/model.pt",
        "model/model_best.pt",
        "model/model_latest.pt"
    ]

    has_model = any(os.path.exists(f) for f in model_files)

    if has_model:
        print("✅ Znaleziono model")
        # Uruchom tryb rozmowy
        print("\n🚀 URUCHAMIAM TRYB ROZMOWY...")

        # Importuj i uruchom
        import sys
        sys.path.append('.')

        try:
            # Dynamically import
            from main import ChatbotConfig, chat_mode

            config = ChatbotConfig(
                learn=False,
                talk=True,
                epochs=1,
                batch=1,
                hidden=256,
                layers=2,
                lr=0.001,
                max_length=200
            )

            chat_mode(config)

        except Exception as e:
            print(f"❌ Błąd: {e}")
            print("\n💡 ALTERNATYWA: Uruchom prosty test")
            subprocess.run(["python3", "test_chat.py"])

    else:
        print("❌ Brak wytrenowanego modelu")
        print("\n🎯 NAJPIERW WYTRENUJ MODEL:")
        print("  python3 main.py --learn")
        print("\nAlbo użyj domyślnych danych testowych:")

        choice = input("Uruchomić trening na danych testowych? (t/n): ")
        if choice.lower() == 't':
            print("Uruchamiam trening...")
            subprocess.run(["python3", "main.py", "--learn"])


if __name__ == "__main__":
    main()