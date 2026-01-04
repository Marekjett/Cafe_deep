import json
import glob
import os
from collections import Counter


def analyze_json_data(folder_path="data/"):
    """
    Analizuje wszystkie pliki JSON w folderze i zlicza wpisy.

    Args:
        folder_path (str): Ścieżka do folderu z danymi
    """

    print("=" * 60)
    print("📊 ANALIZA DANYCH JSON")
    print("=" * 60)

    # Znajdź wszystkie pliki JSON
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        print(f"❌ Nie znaleziono plików JSON w folderze: {folder_path}")
        return

    print(f"📁 Znaleziono {len(json_files)} plików JSON:")
    for i, file in enumerate(json_files, 1):
        print(f"  {i}. {os.path.basename(file)}")

    print("\n" + "=" * 60)

    total_entries = 0
    total_characters = 0
    all_inputs = []
    all_outputs = []
    entry_lengths = []
    problematic_files = []

    # Analizuj każdy plik
    for file_idx, file_path in enumerate(json_files, 1):
        print(f"\n📄 Analizuję plik {file_idx}/{len(json_files)}: {os.path.basename(file_path)}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"  ⚠️  Plik nie zawiera listy, pomijam...")
                problematic_files.append((file_path, "Nie jest listą"))
                continue

            file_entries = len(data)
            total_entries += file_entries

            print(f"  • Wpisy: {file_entries}")

            # Analizuj każdy wpis
            valid_entries = 0
            for i, entry in enumerate(data):
                if isinstance(entry, dict):
                    # Pobierz tekst - zabezpieczenie przed None
                    input_text = entry.get("input")
                    output_text = entry.get("output")

                    # Sprawdź czy teksty istnieją i są stringami
                    if input_text is None or output_text is None:
                        print(f"  ⚠️  Wpis {i}: brak 'input' lub 'output'")
                        continue

                    if not isinstance(input_text, str) or not isinstance(output_text, str):
                        print(f"  ⚠️  Wpis {i}: 'input' lub 'output' nie jest tekstem")
                        continue

                    # Konwertuj na string jeśli potrzeba
                    input_text = str(input_text).strip()
                    output_text = str(output_text).strip()

                    # Pomijaj puste
                    if not input_text or not output_text:
                        print(f"  ⚠️  Wpis {i}: pusty tekst")
                        continue

                    # Zapisz teksty do analizy
                    all_inputs.append(input_text)
                    all_outputs.append(output_text)

                    # Zlicz znaki
                    input_len = len(input_text)
                    output_len = len(output_text)
                    total_len = input_len + output_len
                    total_characters += total_len
                    entry_lengths.append(total_len)

                    valid_entries += 1

                else:
                    print(f"  ⚠️  Wpis {i} nie jest słownikiem")

            print(f"  • Poprawne wpisy: {valid_entries}/{file_entries}")

        except json.JSONDecodeError as e:
            print(f"  ❌ Błąd JSON w pliku {file_path}: {e}")
            problematic_files.append((file_path, f"Błąd JSON: {e}"))
        except Exception as e:
            print(f"  ❌ Błąd odczytu pliku {file_path}: {e}")
            problematic_files.append((file_path, f"Błąd: {e}"))

    print("\n" + "=" * 60)
    print("📈 PODSUMOWANIE STATYSTYK")
    print("=" * 60)

    print(f"\n📊 PODSTAWOWE STATYSTYKI:")
    print(f"  • Łączna liczba wpisów: {total_entries:,}")
    print(f"  • Liczba plików: {len(json_files)}")
    print(f"  • Średnia wpisów na plik: {total_entries / len(json_files):.1f}")

    if problematic_files:
        print(f"\n⚠️  PROBLEMATYCZNE PLIKI:")
        for file, reason in problematic_files:
            print(f"  • {os.path.basename(file)}: {reason}")

    if total_entries > 0:
        print(f"\n📏 DŁUGOŚĆ TEKSTU:")
        print(f"  • Łączna liczba znaków: {total_characters:,}")
        print(f"  • Średnia znaków na wpis: {total_characters / total_entries:.1f}")

        # Oblicz statystyki długości
        if entry_lengths:
            avg_len = sum(entry_lengths) / len(entry_lengths)
            max_len = max(entry_lengths)
            min_len = min(entry_lengths)

            print(f"\n  • Najdłuższy dialog: {max_len:,} znaków")
            print(f"  • Najkrótszy dialog: {min_len:,} znaków")
            print(f"  • Średnia długość: {avg_len:.1f} znaków")

        # Analizuj długości input/output
        if all_inputs and all_outputs:
            try:
                avg_input_len = sum(len(t) for t in all_inputs) / len(all_inputs)
                avg_output_len = sum(len(t) for t in all_outputs) / len(all_outputs)

                print(f"\n  • Średnia input: {avg_input_len:.1f} znaków")
                print(f"  • Średnia output: {avg_output_len:.1f} znaków")
                print(f"  • Stosunek output/input: {avg_output_len / avg_input_len:.2f}")
            except Exception as e:
                print(f"\n  ⚠️  Błąd analizy długości: {e}")

        # Analizuj unikalne słowa
        try:
            all_text = " ".join(all_inputs + all_outputs)
            words = all_text.split()
            unique_words = set(words)

            print(f"\n🔤 ANALIZA TEKSTU:")
            print(f"  • Łącznie słów: {len(words):,}")
            print(f"  • Unikalnych słów: {len(unique_words):,}")
            print(f"  • Stosunek unikalnych/wszystkich: {len(unique_words) / len(words) * 100:.1f}%")

            # Najczęstsze słowa
            if words:
                word_counter = Counter(words)
                most_common = word_counter.most_common(15)

                print(f"\n📈 15 NAJCZĘSTSZYCH SŁÓW:")
                for i, (word, count) in enumerate(most_common, 1):
                    percentage = (count / len(words)) * 100
                    print(f"  {i:2d}. '{word}': {count:,} ({percentage:.2f}%)")

        except Exception as e:
            print(f"\n  ⚠️  Błąd analizy tekstu: {e}")

    print(f"\n{'=' * 60}")
    print("💡 ZALECENIA DLA TRENINGU")
    print(f"{'=' * 60}")

    if total_entries > 0:
        # Zalecenia dotyczące modelu
        if total_entries < 100:
            print(f"\n⚠️  BARDZO MAŁO DANYCH ({total_entries} wpisów)")
            print(f"   Zalecany mały model:")
            print(f"   • hidden: 32-64")
            print(f"   • layers: 1")
            print(f"   • max_length: 100-150")
            print(f"   • epochs: 100+ (do zapamiętania)")

        elif total_entries < 1000:
            print(f"\n📊 UMIARKOWANA ILOŚĆ DANYCH ({total_entries} wpisów)")
            print(f"   Zalecany średni model:")
            print(f"   • hidden: 64-128")
            print(f"   • layers: 1-2")
            print(f"   • max_length: 150-250")
            print(f"   • epochs: 50-100")

        elif total_entries < 10000:
            print(f"\n✅ DOBRA ILOŚĆ DANYCH ({total_entries} wpisów)")
            print(f"   Zalecany większy model:")
            print(f"   • hidden: 128-256")
            print(f"   • layers: 2-3")
            print(f"   • max_length: 250-350")
            print(f"   • epochs: 30-50")

        else:
            print(f"\n🎉 DUŻA ILOŚĆ DANYCH ({total_entries} wpisów)")
            print(f"   Zalecany duży model:")
            print(f"   • hidden: 256-512")
            print(f"   • layers: 3-4")
            print(f"   • max_length: 500")
            print(f"   • epochs: 20-30")

        # Zalecenia dotyczące danych
        print(f"\n📝 ZALECENIA DOTYCZĄCE DANYCH:")

        # Sprawdź czy mamy statystyki długości
        if 'avg_len' in locals():
            if avg_len > 500:
                print(f"   • Rozważ skrócenie długich dialogów (>500 znaków)")
            elif avg_len < 50:
                print(f"   • Dialogi są bardzo krótkie, rozważ rozszerzenie")

        # Stosunek input/output
        if 'avg_input_len' in locals() and 'avg_output_len' in locals():
            try:
                ratio = avg_output_len / avg_input_len
                if ratio < 0.5:
                    print(f"   • Odpowiedzi są bardzo krótkie (stosunek {ratio:.2f})")
                elif ratio > 2:
                    print(f"   • Odpowiedzi są bardzo długie (stosunek {ratio:.2f})")
            except:
                pass

    print(f"\n{'=' * 60}")

    return {
        "total_entries": total_entries,
        "total_files": len(json_files),
        "total_characters": total_characters,
        "average_entry_length": total_characters / total_entries if total_entries > 0 else 0,
        "words_count": len(words) if 'words' in locals() else 0,
        "unique_words": len(unique_words) if 'unique_words' in locals() else 0,
        "problematic_files": problematic_files
    }


def fix_problematic_json(file_path):
    """
    Próbuje naprawić uszkodzony plik JSON
    """
    print(f"\n🔧 Próbuję naprawić: {os.path.basename(file_path)}")

    try:
        # Wczytaj jako tekst
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Proste naprawy JSON
        content = content.strip()

        # Jeśli zaczyna się od { ale nie jest listą, dodaj []
        if content.startswith('{') and not content.startswith('[{'):
            content = '[' + content + ']'

        # Zamień pojedyncze cudzysłowy na podwójne
        content = content.replace("'", '"')

        # Napraw brakujące przecinki
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith('//'):
                # Dodaj przecinek jeśli linia kończy się } a następna zaczyna od {
                if line.endswith('}') and i + 1 < len(lines) and lines[i + 1].strip().startswith('{'):
                    line += ','
                fixed_lines.append(line)

        fixed_content = '\n'.join(fixed_lines)

        # Zapraw naprawiony plik
        backup_path = file_path + '.backup'
        os.rename(file_path, backup_path)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        print(f"  ✅ Utworzono kopię: {os.path.basename(backup_path)}")
        print(f"  ✅ Naprawiono plik: {os.path.basename(file_path)}")
        return True

    except Exception as e:
        print(f"  ❌ Nie udało się naprawić: {e}")
        return False


def generate_data_report(stats, output_file="data_report.txt"):
    """Generuje raport z analizy danych do pliku"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("📊 RAPORT ANALIZY DANYCH\n")
        f.write("=" * 60 + "\n\n")

        f.write("PODSTAWOWE STATYSTYKI:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Łączna liczba wpisów: {stats['total_entries']:,}\n")
        f.write(f"Liczba plików JSON: {stats['total_files']}\n")
        f.write(f"Łączna liczba znaków: {stats['total_characters']:,}\n")
        f.write(f"Średnia znaków na wpis: {stats['average_entry_length']:.1f}\n")

        if stats.get('words_count', 0) > 0:
            f.write(f"\nANALIZA TEKSTU:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Łącznie słów: {stats['words_count']:,}\n")
            f.write(f"Unikalnych słów: {stats['unique_words']:,}\n")
            f.write(f"Współczynnik różnorodności: {stats['unique_words'] / stats['words_count'] * 100:.1f}%\n")

        if stats.get('problematic_files'):
            f.write(f"\n⚠️  PROBLEMATYCZNE PLIKI:\n")
            f.write("-" * 40 + "\n")
            for file, reason in stats['problematic_files']:
                f.write(f"• {os.path.basename(file)}: {reason}\n")

    print(f"\n💾 Raport zapisano do: {output_file}")


# Funkcja pomocnicza do szybkiego sprawdzenia
def quick_count(folder_path="data/"):
    """Szybkie zliczenie wpisów bez szczegółowej analizy"""

    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    total = 0

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    total += len(data)
        except:
            continue

    print(f"\n⚡ SZYBKIE ZLICZENIE:")
    print(f"   Pliki JSON: {len(json_files)}")
    print(f"   Wpisy: {total}")
    return total


if __name__ == "__main__":
    print("🔍 ANALIZATOR DANYCH JSON")
    print("Sprawdzam folder 'data/'...\n")

    # Sprawdź czy folder istnieje
    if not os.path.exists("data/"):
        print("❌ Folder 'data/' nie istnieje!")
        print("Tworzę folder 'data/'...")
        os.makedirs("data/", exist_ok=True)
        print("✅ Utworzono folder 'data/'")
        print("Dodaj pliki JSON do folderu 'data/' i uruchom ponownie.")
    else:
        # Uruchom pełną analizę
        stats = analyze_json_data("data/")

        # Zapisz raport jeśli są dane
        if stats["total_entries"] > 0:
            generate_data_report(stats)

            # Sprawdź czy są problematyczne pliki
            if stats.get('problematic_files'):
                print(f"\n🛠️  CHCESZ SPRÓBOWAĆ NAPRAWIĆ PROBLEMATYCZNE PLIKI?")
                print(f"   Znaleziono {len(stats['problematic_files'])} plików z błędami")

                choice = input("   Naprawić automatycznie? (t/n): ").strip().lower()
                if choice == 't':
                    for file_path, reason in stats['problematic_files']:
                        fix_problematic_json(file_path)
                    print("\n✅ Próba naprawy zakończona.")
                    print("   Uruchom analizę ponownie by sprawdzić efekty.")

            # Dodatkowe opcje
            print("\n🎯 DODATKOWE OPCJE:")
            print("1. Szybkie zliczenie")
            print("2. Analizuj inny folder")
            print("3. Wyjście")

            choice = input("\nWybierz opcję (1-3): ").strip()

            if choice == "1":
                quick_count("data/")
            elif choice == "2":
                folder = input("Podaj ścieżkę do folderu: ").strip()
                if os.path.exists(folder):
                    analyze_json_data(folder)
                else:
                    print(f"❌ Folder '{folder}' nie istnieje!")