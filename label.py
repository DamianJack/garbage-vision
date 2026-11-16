from pathlib import Path
from collections import Counter
from collections import defaultdict
import random

counts = Counter()

groups = defaultdict(list)   

for file in Path("GarbageDataset/labels").glob("*.txt"):
    image_classes = set()
    for line in file.read_text().splitlines():
        if not line.strip():
            continue
        image_classes.add( int(line.split()[0]) )    

    if image_classes:                      
        rarest = min(image_classes, key=lambda c: counts[c])
    else:                                 
        rarest = -1                          

    groups[rarest].append(file.stem)

for g, items in sorted(groups.items()):
    print(g, len(items))

random.seed( 42 )                 
for g, items in groups.items():
    random.shuffle( items )        

print(groups[4][:3])

train, val, test = [], [], []     

for g, items in groups.items():
    n = len(items)
    n_train = int(n * 0.7)              
    n_val   = int(n * 0.15)            

    train += items[ : n_train ]
    val  += items[ n_train : n_train + n_val ]
    test += items[ n_train + n_val : ]
print(len(train), len(val), len(test))

name = train[0]
image_file = list(Path("GarbageDataset/images").glob(name + ".*"))[0]
print(image_file)