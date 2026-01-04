import os

def folder_size(folder):
    total_size = 0
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path):
            total_size += os.path.getsize(path)
    return total_size / 1024 / 1024  # MB

books_folder = '/home/neon/PycharmProjects/Cafembler/books/'
data_folder = '/home/neon/PycharmProjects/Cafembler/data/'

books_size = folder_size(books_folder)
data_size = folder_size(data_folder)

print(f"📚 books/: {books_size:.2f} MB")
print(f"🗂 data/: {data_size:.2f} MB")
print(f"💾 Łącznie: {books_size + data_size:.2f} MB")
