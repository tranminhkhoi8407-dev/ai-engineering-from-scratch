from datasets import load_dataset

# Stream dữ liệu để tránh tải toàn bộ C4 (rất lớn) về máy.
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

# Data chỉ được fetch khi bạn iterate
for i, row in enumerate(ds):
    print(row["text"][:100])
    if i == 4:
        break   # dừng sớm -> chỉ tải đúng vài row đầu