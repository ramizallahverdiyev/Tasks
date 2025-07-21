import sys

current_city = None
total_age = 0
count = 0

for line in sys.stdin:
    city, age = line.strip().split('\t')
    age = int(age)

    if city != current_city and current_city is not None:
        avg_age = total_age / count
        print(f"{current_city}\tAverage Age: {avg_age:.2f}")
        total_age = 0
        count = 0

    current_city = city
    total_age += age
    count += 1

if current_city:
    avg_age = total_age / count
    print(f"{current_city}\tAverage Age: {avg_age:.2f}")
