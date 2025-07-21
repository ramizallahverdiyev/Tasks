import sys

for line in sys.stdin:
    gender, projects = line.strip().split('\t')
    print(f"{gender}\t{projects}")
