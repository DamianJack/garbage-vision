from collections import Counter
from pathlib import Path

OUT = Path("GarbageSplit")

for split_name in ["train", "val", "test"]:
    counts = Counter()

    # same counting code as before, just a different folder
    for file in (OUT / "labels" / split_name).glob("*.txt"):
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            cls = int(line.split()[0])
            counts[cls] += 1

    total = sum(counts.values())
    print(f"\n{split_name}  (total boxes: {total})")

    for cls in sorted(counts):
        percent = counts[cls] / total * 100  
        print(f"  class {cls}: {counts[cls]:>6}  = {percent:.1f}%")