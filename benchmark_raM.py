import torch
import os
import psutil
import time

process = psutil.Process(os.getpid())

def ram_mb():
    return process.memory_info().rss / 1024**2


class DummyLSTM(torch.nn.Module):
    def __init__(self, vocab_size, hidden, layers):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, hidden)
        self.lstm = torch.nn.LSTM(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm(x)
        return self.fc(x)


VOCAB = 50_000
SEQ = 2000
BATCH = 1

print("=== TEST MAKSYMALNEGO MODELU (RAM ONLY) ===")
print("Start RAM:", f"{ram_mb():.1f} MB")

configs = [
    (256, 4),
    (512, 4),
    (512, 8),
    (768, 8),
    (1024, 8),
    (1024, 12),
    (1024, 16),
]

for hidden, layers in configs:
    try:
        print(f"\n➡️  Test: hidden={hidden}, layers={layers}")
        before = ram_mb()

        model = DummyLSTM(VOCAB, hidden, layers)
        x = torch.randint(0, VOCAB, (BATCH, SEQ))
        y = model(x)

        after = ram_mb()
        print(f"   RAM +{after - before:.1f} MB | razem {after:.1f} MB")

        del model, x, y
        torch.cuda.empty_cache()
        time.sleep(1)

    except RuntimeError as e:
        print("❌ OOM / FAIL:", e)
        break
