from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["company_db"]
employees = db["employees"]

employee_data = [
    {"employee_id": 1, "first_name": "Ali", "last_name": "Veli", "salary": 2500, "email": "ali@example.com", "job_id": "DEV001"},
    {"employee_id": 2, "first_name": "Aysel", "last_name": "Hesenli", "salary": 3000, "email": "aysel@example.com", "job_id": "HR002"},
    {"employee_id": 3, "first_name": "Murad", "last_name": "Memmedov", "salary": 2000, "email": "murad@example.com", "job_id": "ACC003"}
]
employees.insert_many(employee_data)

print("Bütün əməkdaşlar:")
for emp in employees.find():
    print(emp)

min_salary = employees.find_one(sort=[("salary", 1)])
print("\nƏn az maaş alan əməkdaş:")
print(min_salary)

print("\nMaaşa görə azalan sırada əməkdaşlar:")
for emp in employees.find().sort("salary", -1):
    print(emp)
