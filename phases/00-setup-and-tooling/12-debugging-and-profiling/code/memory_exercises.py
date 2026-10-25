import tracemalloc

def load_data():
    # Giả lập đoạn load data của bạn
    texts = ["hello world " * 100 for _ in range(50_000)]        # dòng A: list string lớn
    labels = [i % 10 for i in range(500_000)]                    # dòng B: list số nhỏ hơn
    return texts, labels

tracemalloc.start()

data = load_data()

snapshot = tracemalloc.take_snapshot()
tracemalloc.stop()

print("Top 5 memory allocations:")
for stat in snapshot.statistics("lineno")[:5]:
    print(stat)