import json
import os
import glob
import re
from collections import Counter
import math


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Wyświetl pasek postępu"""
    percent = f"{100 * (iteration / float(total)):5.1f}%"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '░' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent} {suffix}', end='', flush=True)
    if iteration == total:
        print()


def check_json_files(folder="datas/"):
    """Sprawdź poprawność plików JSON"""
    print(f"\n{'=' * 60}")
    print("🔍 SPRAWDZANIE PLIKÓW JSON")
    print(f"{'=' * 60}")

    data_files = glob.glob(os.path.join(folder, "*.json"))
    print(f"Znaleziono {len(data_files)} plików JSON w folderze {folder}")

    if not data_files:
        print("❌ Brak plików JSON!")
        return False

    all_data = []
    errors = []
    stats = {
        "total_files": len(data_files),
        "valid_files": 0,
        "total_dialogs": 0,
        "valid_dialogs": 0,
        "avg_input_len": 0,
        "avg_output_len": 0
    }

    for i, file in enumerate(data_files):
        print_progress_bar(i, len(data_files),
                           prefix=f"Sprawdzanie {i + 1}/{len(data_files)}:",
                           suffix=os.path.basename(file)[:20],
                           length=30)

        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                errors.append(f"{file}: Główna struktura nie jest listą")
                continue

            file_dialogs = 0
            file_valid = 0

            for j, item in enumerate(data):
                if not isinstance(item, dict):
                    errors.append(f"{file}[{j}]: Element nie jest słownikiem")
                    continue

                input_text = item.get("input", "").strip()
                output_text = item.get("output", "").strip()

                # Sprawdź czy pola istnieją
                if not input_text:
                    errors.append(f"{file}[{j}]: Brak pola 'input' lub jest pusty")
                    continue
                if not output_text:
                    errors.append(f"{file}[{j}]: Brak pola 'output' lub jest pusty")
                    continue

                # Sprawdź długość
                if len(input_text) < 5:
                    errors.append(f"{file}[{j}]: Input za krótki ({len(input_text)} znaków)")
                    continue
                if len(output_text) < 5:
                    errors.append(f"{file}[{j}]: Output za krótki ({len(output_text)} znaków)")
                    continue
                if len(input_text) > 500:
                    errors.append(f"{file}[{j}]: Input za długi ({len(input_text)} znaków)")
                    continue
                if len(output_text) > 500:
                    errors.append(f"{file}[{j}]: Output za długi ({len(output_text)} znaków)")
                    continue

                file_valid += 1
                all_data.append({
                    "input": input_text,
                    "output": output_text,
                    "file": os.path.basename(file),
                    "index": j
                })

            stats["total_dialogs"] += len(data)
            stats["valid_dialogs"] += file_valid

            if file_valid > 0:
                stats["valid_files"] += 1

        except json.JSONDecodeError as e:
            errors.append(f"{file}: Błąd JSON - {e}")
        except Exception as e:
            errors.append(f"{file}: Nieznany błąd - {e}")

    print_progress_bar(len(data_files), len(data_files),
                       prefix="Sprawdzanie zakończone",
                       suffix=f"{stats['valid_dialogs']} poprawnych dialogów",
                       length=30)

    # Oblicz statystyki długości
    if all_data:
        avg_input = sum(len(d["input"]) for d in all_data) / len(all_data)
        avg_output = sum(len(d["output"]) for d in all_data) / len(all_data)
        stats["avg_input_len"] = round(avg_input, 1)
        stats["avg_output_len"] = round(avg_output, 1)

    # Wyświetl podsumowanie
    print(f"\n{'=' * 60}")
    print("📊 PODSUMOWANIE")
    print(f"{'=' * 60}")
    print(f"📁 Pliki: {stats['valid_files']}/{stats['total_files']} poprawnych")
    print(f"💬 Dialogi: {stats['valid_dialogs']}/{stats['total_dialogs']} poprawnych")
    print(f"📏 Średnia długość input: {stats['avg_input_len']} znaków")
    print(f"📏 Średnia długość output: {stats['avg_output_len']} znaków")

    if all_data:
        # Analiza słów kluczowych
        print(f"\n{'─' * 50}")
        print("🔤 ANALIZA SŁÓW KLUCZOWYCH")
        print(f"{'─' * 50}")

        all_inputs = " ".join(d["input"].lower() for d in all_data)
        all_outputs = " ".join(d["output"].lower() for d in all_data)

        # Najczęstsze słowa w input
        words_input = re.findall(r'\b\w{3,}\b', all_inputs)
        freq_input = Counter(words_input)

        # Najczęstsze słowa w output
        words_output = re.findall(r'\b\w{3,}\b', all_outputs)
        freq_output = Counter(words_output)

        print("Najczęstsze słowa w pytaniach (input):")
        for word, count in freq_input.most_common(10):
            print(f"  {word}: {count}")

        print("\nNajczęstsze słowa w odpowiedziach (output):")
        for word, count in freq_output.most_common(10):
            print(f"  {word}: {count}")

    # Wyświetl błędy jeśli istnieją
    if errors:
        print(f"\n{'─' * 50}")
        print("⚠️  BŁĘDY")
        print(f"{'─' * 50}")
        for error in errors[:20]:  # Pokaż tylko pierwsze 20 błędów
            print(f"❌ {error}")
        if len(errors) > 20:
            print(f"... i {len(errors) - 20} więcej błędów")

    return len(all_data) > 0


def check_text_files(folder="books/"):
    """Sprawdź pliki tekstowe (książki)"""
    print(f"\n{'=' * 60}")
    print("📚 SPRAWDZANIE PLIKÓW TEKSTOWYCH")
    print(f"{'=' * 60}")

    text_files = glob.glob(os.path.join(folder, "*.txt"))
    print(f"Znaleziono {len(text_files)} plików TXT w folderze {folder}")

    if not text_files:
        print("ℹ️  Brak plików tekstowych")
        return True

    stats = {
        "total_files": len(text_files),
        "valid_files": 0,
        "total_chars": 0,
        "total_words": 0
    }

    for i, file in enumerate(text_files):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                print(f"❌ {os.path.basename(file)}: Plik pusty")
                continue

            # Statystyki
            chars = len(content)
            words = len(content.split())
            lines = len(content.split('\n'))

            stats["total_chars"] += chars
            stats["total_words"] += words
            stats["valid_files"] += 1

            print(f"✅ {os.path.basename(file):30} | {chars:8,} znaków | {words:6,} słów | {lines:4} linii")

        except UnicodeDecodeError:
            print(f"❌ {os.path.basename(file)}: Problem z kodowaniem UTF-8")
        except Exception as e:
            print(f"❌ {os.path.basename(file)}: Błąd - {e}")

    if stats["valid_files"] > 0:
        print(f"\n{'─' * 50}")
        print("📊 STATYSTYKI KSIĄŻEK")
        print(f"{'─' * 50}")
        print(f"📁 Pliki: {stats['valid_files']}/{stats['total_files']} poprawnych")
        print(f"📝 Znaki: {stats['total_chars']:,}")
        print(f"🔤 Słowa: {stats['total_words']:,}")
        print(f"📏 Średnio na plik: {stats['total_chars'] // stats['valid_files']:,} znaków")

    return stats["valid_files"] > 0


def check_model_files(model_dir="model/", backup_dir="backup_model/"):
    """Sprawdź istniejące modele"""
    print(f"\n{'=' * 60}")
    print("🤖 SPRAWDZANIE MODELI")
    print(f"{'=' * 60}")

    import torch

    # Sprawdź foldery
    for dir_name, dir_path in [("Model", model_dir), ("Backup", backup_dir)]:
        if os.path.exists(dir_path):
            files = glob.glob(os.path.join(dir_path, "*.pt"))
            print(f"{dir_name}: {len(files)} plików .pt")

            # Sprawdź rozmiary plików
            for file in files[:5]:  # Pokaż pierwsze 5
                try:
                    size = os.path.getsize(file)
                    print(f"  📄 {os.path.basename(file):30} | {size / 1024 / 1024:6.1f} MB")
                except:
                    print(f"  ❌ {os.path.basename(file)}: Nie można odczytać rozmiaru")

            if len(files) > 5:
                print(f"  ... i {len(files) - 5} więcej plików")
        else:
            print(f"{dir_name}: Folder nie istnieje")

    # Sprawdź czy można załadować główny model
    main_model = os.path.join(model_dir, "model.pt")
    if os.path.exists(main_model):
        print(f"\n{'─' * 50}")
        print("🔍 TEST ŁADOWANIA GŁÓWNEGO MODELU")
        print(f"{'─' * 50}")

        try:
            checkpoint = torch.load(main_model, map_location='cpu', weights_only=False)
            print("✅ Model załadowany poprawnie!")

            # Wyświetl informacje o modelu
            print(f"📅 Data: {checkpoint.get('timestamp', 'Nieznana')}")
            print(f"🎯 Epoka: {checkpoint.get('epoch', 'Nieznana')}")
            print(f"📉 Val loss: {checkpoint.get('val_loss', 'Nieznana')}")
            print(f"🔤 Słownik: {len(checkpoint.get('stoi', {}))} tokenów")
            print(f"🧠 Hidden size: {checkpoint.get('hidden_size', 'Nieznany')}")
            print(f"🏗️  Layers: {checkpoint.get('num_layers', 'Nieznane')}")

        except Exception as e:
            print(f"❌ Błąd ładowania modelu: {e}")
    else:
        print("ℹ️  Główny model nie istnieje")


def generate_sample_data(num_samples=10):
    """Wygeneruj przykładowe dane dla testów"""
    print(f"\n{'=' * 60}")
    print("🎲 PRZYKŁADOWE DANE Z DATASETU")
    print(f"{'=' * 60}")

    data_files = glob.glob("data/*.json")
    if not data_files:
        print("❌ Brak plików z danymi")
        return

    all_samples = []

    for file in data_files[:3]:  # Sprawdź tylko pierwsze 3 pliki
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data[:5]:  # Pierwsze 5 z każdego pliku
                    if isinstance(item, dict) and "input" in item and "output" in item:
                        all_samples.append({
                            "input": item["input"][:100] + "..." if len(item["input"]) > 100 else item["input"],
                            "output": item["output"][:100] + "..." if len(item["output"]) > 100 else item["output"],
                            "file": os.path.basename(file)
                        })
        except:
            continue

    if all_samples:
        print(f"Pokazuję {min(num_samples, len(all_samples))} przykładowych dialogów:\n")
        for i, sample in enumerate(all_samples[:num_samples]):
            print(f"{i + 1}. [{sample['file']}]")
            print(f"   ❓ {sample['input']}")
            print(f"   💬 {sample['output']}")
            print()
    else:
        print("❌ Nie znaleziono poprawnych dialogów")


def check_vocabulary_size(data_folder="data/"):
    """Sprawdź rozmiar słownika jaki zostanie wygenerowany"""
    print(f"\n{'=' * 60}")
    print("🔠 ESTYMACJA ROZMIARU SŁOWNIKA")
    print(f"{'=' * 60}")

    data_files = glob.glob(os.path.join(data_folder, "*.json"))
    if not data_files:
        print("❌ Brak plików z danymi")
        return

    all_text = ""
    char_counts = Counter()

    for file in data_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    if isinstance(item, dict) and "input" in item and "output" in item:
                        text = item["input"] + " " + item["output"]
                        all_text += text + " "
                        char_counts.update(text)
        except:
            continue

    if not all_text:
        print("❌ Nie udało się zebrać tekstu")
        return

    # Statystyki
    total_chars = len(all_text)
    unique_chars = len(char_counts)

    # Próg częstotliwości
    min_freq = 2
    common_chars = [c for c, count in char_counts.items() if count >= min_freq or c in ' .,!?\nąćęłńóśźżĄĆĘŁŃÓŚŹŻ']

    print(f"📊 Statystyki tekstu:")
    print(f"  • Znaki łącznie: {total_chars:,}")
    print(f"  • Unikalne znaki: {unique_chars}")
    print(f"  • Częste znaki (≥{min_freq}): {len(common_chars)}")
    print(f"  • Znaki specjalne: 4 (<BOS>, <EOS>, <PAD>, <UNK>)")
    print(f"  • Przewidywany rozmiar słownika: {len(common_chars) + 4}")
    print(f"\n📈 Rozkład najczęstszych znaków:")
    for char, count in char_counts.most_common(20):
        perc = (count / total_chars) * 100
        print(f"  '{char if char != ' ' else '[SPACE]'}': {count:6,} ({perc:.1f}%)")

    # Sprawdź polskie znaki
    polish_chars = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
    polish_count = sum(char_counts[char] for char in polish_chars if char in char_counts)
    print(f"\n🇵🇱 Polskie znaki: {polish_count:,} ({polish_count / total_chars * 100:.1f}%)")


def quick_test_model():
    """Szybki test działania modelu"""
    print(f"\n{'=' * 60}")
    print("⚡ SZYBKI TEST MODELU")
    print(f"{'=' * 60}")

    import torch
    from datetime import datetime

    # Sprawdź czy PyTorch działa
    print("✅ PyTorch wersja:", torch.__version__)
    print("✅ CUDA dostępne:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Pamięć GPU: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Test prostego tensora
    print("\n🧪 Test tensora PyTorch:")
    x = torch.randn(3, 3)
    print(f"   Tensor 3x3:\n{x}")
    print(f"   Rozmiar: {x.shape}")

    # Test czasu
    print("\n⏱️  Test wydajności:")
    start = datetime.now()

    # Prosta operacja na GPU jeśli dostępne
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    for _ in range(10):
        c = torch.matmul(a, b)

    end = datetime.now()
    elapsed = (end - start).total_seconds()

    print(f"   Czas 10 mnożeń macierzy 1000x1000: {elapsed:.3f}s")
    print(f"   Urządzenie: {device}")

    # Sprawdź pamięć
    if torch.cuda.is_available():
        print(f"\n💾 Pamięć GPU:")
        print(f"   Zajęta: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
        print(f"   Zarezerwowana: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")


def check_folder_structure():
    """Sprawdź strukturę folderów projektu"""
    print(f"\n{'=' * 60}")
    print("📁 STRUKTURA PROJEKTU")
    print(f"{'=' * 60}")

    folders = ["data", "books", "model", "backup_model"]
    current_dir = os.getcwd()

    print(f"📂 Aktualny folder: {current_dir}")
    print(f"\nStruktura:")

    for folder in folders:
        path = os.path.join(current_dir, folder)
        if os.path.exists(path):
            files = os.listdir(path)
            files_count = len(files)
            print(f"  ✅ {folder}/ - {files_count} plików")

            # Pokaż przykładowe pliki
            if files_count > 0:
                sample_files = files[:3]
                for file in sample_files:
                    full_path = os.path.join(path, file)
                    if os.path.isfile(full_path):
                        size = os.path.getsize(full_path)
                        print(f"      📄 {file} - {size / 1024:.1f} KB")
        else:
            print(f"  ❌ {folder}/ - BRAK")


def main():
    """Główna funkcja sprawdzająca"""
    print(f"{'=' * 80}")
    print("🧪 TRY_LEARN - SYSTEM SPRAWDZANIA POPRAWNOŚCI")
    print(f"{'=' * 80}")
    print("Skrypt do szybkiego testowania poprawności danych i przygotowania do treningu")
    print(f"{'=' * 80}")

    # Sprawdź strukturę folderów
    check_folder_structure()

    # Sprawdź pliki JSON
    json_ok = check_json_files("data/")

    # Sprawdź pliki tekstowe
    books_ok = check_text_files("books/")

    # Sprawdź istniejące modele
    check_model_files("model/", "backup_model/")

    # Estymuj rozmiar słownika
    check_vocabulary_size("data/")

    # Wygeneruj przykładowe dane
    generate_sample_data(8)

    # Szybki test modelu
    quick_test_model()

    # Podsumowanie
    print(f"\n{'=' * 80}")
    print("🎯 PODSUMOWANIE I NASTĘPNE KROKI")
    print(f"{'=' * 80}")

    if json_ok:
        print("✅ Pliki JSON są poprawne - możesz rozpocząć trening")
        print("   Uruchom: python main.py --train")
    else:
        print("❌ Problem z plikami JSON - popraw je przed treningiem")

    if books_ok:
        print("✅ Pliki tekstowe są poprawne - będą użyte w treningu")
    else:
        print("ℹ️  Brak plików tekstowych, ale to nie jest wymagane")

    print(f"\n📋 Zalecane kroki:")
    print("1. Sprawdź czy masz wystarczająco danych (minimum 1000 dialogów)")
    print("2. Upewnij się że dane są zróżnicowane tematycznie")
    print("3. Usuń duplikaty z danych")
    print("4. Dostosuj parametry w ChatbotConfig jeśli potrzeba")
    print("5. Uruchom trening: python main.py --train")
    print("6. Przetestuj model: python main.py --talk")

    print(f"\n{'=' * 80}")
    print("🚀 Gotowy do działania!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()