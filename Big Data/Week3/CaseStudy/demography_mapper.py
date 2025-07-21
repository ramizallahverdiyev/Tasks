import sys

for line in sys.stdin:
    parts = line.strip().split('\t')
    if len(parts) == 3:
        name, age, city = parts
        print(f"{city}\t{age}")
