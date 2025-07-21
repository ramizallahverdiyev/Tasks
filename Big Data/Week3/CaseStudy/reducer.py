import sys

current_gender = None
projects_list = []

for line in sys.stdin:
    gender, count = line.strip().split('\t')
    count = int(count)

    if gender != current_gender and current_gender is not None:
        print(f"{current_gender}\tMin: {min(projects_list)}, Max: {max(projects_list)}")
        projects_list = []

    current_gender = gender
    projects_list.append(count)

if current_gender:
    print(f"{current_gender}\tMin: {min(projects_list)}, Max: {max(projects_list)}")
