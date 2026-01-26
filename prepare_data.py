"""
📊 PREPARE_DATA - Przygotowanie danych z wielu folderów
"""

import os
import re
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import logger, cfg


# ==================== KLASY DO PRZYGOTOWANIA DANYCH ====================
class DataPreparer:
    """Główna klasa do przygotowania danych"""

    def __init__(self):
        self.output_file = Path(cfg.prepared_dir) / "all_data.txt"
        self.metadata_file = Path(cfg.prepared_dir) / "metadata.json"
        self.stats = {
            "total_files": 0,
            "total_samples": 0,
            "total_chars": 0,
            "sources": {},
            "errors": []
        }

    def find_data_folders(self) -> List[Path]:
        """Znajduje wszystkie foldery zaczynające się od 'data_'"""
        current_dir = Path(".")
        data_folders = []

        for item in current_dir.iterdir():
            if item.is_dir() and item.name.startswith("data_"):
                data_folders.append(item)
                logger.info(f"📁 Znaleziono folder: {item.name}")

        # Dodaj też standardowy folder 'data' jeśli istnieje
        if Path("data").exists():
            data_folders.append(Path("data"))

        return data_folders

    def process_file(self, file_path: Path) -> List[str]:
        """Przetwarza pojedynczy plik i zwraca próbki"""
        samples = []

        try:
            if file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Różne strategie dla różnych typów plików
                    if "news" in file_path.name.lower() or "article" in file_path.name.lower():
                        # Dla newsów - podziel na akapity
                        paragraphs = re.split(r'\n\s*\n', content)
                        for para in paragraphs:
                            para = para.strip()
                            if 100 < len(para) < 5000:
                                samples.append(para)

                    elif "code" in file_path.name.lower() or "python" in file_path.name.lower():
                        # Dla kodu - zachowaj całe funkcje
                        lines = content.split('\n')
                        current_chunk = []

                        for line in lines:
                            line = line.strip()
                            if line:
                                current_chunk.append(line)

                                # Jeśli znaleziono koniec funkcji lub duży blok
                                if line.startswith('def ') and len(current_chunk) > 3:
                                    samples.append('\n'.join(current_chunk))
                                    current_chunk = []

                        # Dodaj pozostały chunk
                        if current_chunk and len('\n'.join(current_chunk)) > 50:
                            samples.append('\n'.join(current_chunk))

                    else:
                        # Domyślnie - podziel na linie/akapity
                        lines = content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if 20 < len(line) < 1000:
                                samples.append(line)

            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                samples.append(item)
                            elif isinstance(item, dict):
                                # Konwertuj słownik na tekst
                                text = ' '.join([f"{k}: {v}" for k, v in item.items()])
                                if len(text) > 20:
                                    samples.append(text)

            elif file_path.suffix == '.csv':
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = ' '.join(row.values())
                        if len(text) > 20:
                            samples.append(text)

        except Exception as e:
            self.stats["errors"].append(f"{file_path}: {str(e)}")

        return samples

    def process_folder(self, folder_path: Path) -> Dict[str, Any]:
        """Przetwarza cały folder"""
        folder_stats = {
            "name": folder_path.name,
            "files_processed": 0,
            "samples_found": 0,
            "samples": []
        }

        # Znajdź wszystkie pliki tekstowe
        file_patterns = ['*.txt', '*.json', '*.csv']
        files = []

        for pattern in file_patterns:
            files.extend(list(folder_path.rglob(pattern)))

        logger.info(f"   📂 {folder_path.name}: {len(files)} plików")

        # Przetwarzaj pliki równolegle
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.process_file, file): file for file in files}

            for future in as_completed(futures):
                file = futures[future]
                try:
                    samples = future.result()
                    if samples:
                        folder_stats["samples"].extend(samples)
                        folder_stats["samples_found"] += len(samples)
                        folder_stats["files_processed"] += 1
                except Exception as e:
                    self.stats["errors"].append(f"{file}: {str(e)}")

        return folder_stats

    def prepare_all_data(self) -> None:
        """Główna funkcja przygotowania danych"""
        logger.info("=" * 60)
        logger.info("📊 PRZYGOTOWYWANIE DANYCH Z WSZYSTKICH FOLDERÓW")
        logger.info("=" * 60)

        # Znajdź foldery
        data_folders = self.find_data_folders()

        if not data_folders:
            logger.error("❌ Nie znaleziono folderów zaczynających się od 'data_'")
            return

        all_samples = []

        # Przetwórz każdy folder
        for folder in data_folders:
            logger.info(f"\n🔍 Przetwarzam folder: {folder.name}")

            folder_stats = self.process_folder(folder)

            if folder_stats["samples"]:
                all_samples.extend(folder_stats["samples"])
                self.stats["sources"][folder.name] = folder_stats["samples_found"]

                logger.info(f"   ✅ Znaleziono: {folder_stats['samples_found']} próbek")
                logger.info(f"   📝 Przykłady:")
                for sample in random.sample(folder_stats["samples"], min(3, len(folder_stats["samples"]))):
                    logger.info(f"      • {sample[:80]}...")
            else:
                logger.warning(f"   ⚠️ Brak danych w folderze {folder.name}")

        # Przetasuj i ogranicz
        if all_samples:
            random.shuffle(all_samples)

            # Ogranicz do 1 miliona próbek (dla pamięci)
            if len(all_samples) > 1000000:
                all_samples = all_samples[:1000000]
                logger.warning(f"⚠️ Ograniczono do 1,000,000 próbek")

            # Zapisz do jednego pliku
            self._save_to_file(all_samples)

            # Zapisz metadane
            self._save_metadata()

            # Podsumowanie
            self._print_summary()
        else:
            logger.error("❌ Nie znaleziono żadnych danych!")

    def _save_to_file(self, samples: List[str]) -> None:
        """Zapisuje wszystkie dane do jednego pliku"""
        logger.info(f"\n💾 Zapisuję {len(samples):,} próbek do {self.output_file}")

        with open(self.output_file, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(samples, 1):
                f.write(sample + "\n\n")

                # Progress bar co 10k próbek
                if i % 10000 == 0:
                    logger.info(f"   Zapisano {i:,}/{len(samples):,} próbek")

        logger.info(f"✅ Zapisano wszystkie dane do {self.output_file}")

    def _save_metadata(self) -> None:
        """Zapisuje metadane"""
        metadata = {
            "total_samples": self.stats["total_samples"],
            "total_chars": self.stats["total_chars"],
            "sources": self.stats["sources"],
            "created": os.path.getmtime(str(self.output_file)),
            "file_size": os.path.getsize(self.output_file),
            "errors": self.stats["errors"][:10]  # Tylko 10 pierwszych błędów
        }

        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 Metadane zapisane do {self.metadata_file}")

    def _print_summary(self) -> None:
        """Wyświetla podsumowanie"""
        logger.info("=" * 60)
        logger.info("📈 PODSUMOWANIE PRZYGOTOWANIA DANYCH")
        logger.info("=" * 60)

        total_samples = self.stats["total_samples"]
        total_chars_mb = self.stats["total_chars"] / (1024 * 1024)

        logger.info(f"📊 STATYSTYKI:")
        logger.info(f"   • Całkowite próbki: {total_samples:,}")
        logger.info(f"   • Rozmiar danych: {total_chars_mb:.1f} MB")
        logger.info(f"   • Źródła danych: {len(self.stats['sources'])}")

        logger.info(f"\n📁 ŹRÓDŁA:")
        for source, count in self.stats["sources"].items():
            logger.info(f"   • {source}: {count:,} próbek")

        if self.stats["errors"]:
            logger.warning(f"\n⚠️ BŁĘDY ({len(self.stats['errors'])}):")
            for error in self.stats["errors"][:5]:
                logger.warning(f"   • {error}")

        logger.info(f"\n💾 WYJŚCIE:")
        logger.info(f"   • Dane: {self.output_file}")
        logger.info(f"   • Metadane: {self.metadata_file}")

        logger.info("\n🎮 UŻYCIE:")
        logger.info("   python main.py --train      # Trening na przygotowanych danych")
        logger.info("   python main.py --prepare    # Ponowne przygotowanie danych")
        logger.info("=" * 60)


# ==================== FUNKCJE POMOCNICZE ====================
def clean_text(text: str) -> str:
    """Czyści tekst"""
    # Usuń nadmiarowe białe znaki
    text = re.sub(r'\s+', ' ', text)

    # Usuń specjalne znaki (opcjonalnie)
    # text = re.sub(r'[^\w\s.,!?;:()\-\'"ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', '', text)

    return text.strip()


def split_into_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Dzieli długi tekst na kawałki"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        word_size = len(word) + 1  # +1 dla spacji

        if current_size + word_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# ==================== GŁÓWNA FUNKCJA ====================
def main():
    """Główna funkcja przygotowania danych"""
    preparer = DataPreparer()
    preparer.prepare_all_data()


if __name__ == "__main__":
    main()