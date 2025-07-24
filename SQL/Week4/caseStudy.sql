-- 1) Return the name of the employee with the lowest salary in department 90
SELECT first_name, last_name
FROM employees
WHERE department_id = 90
ORDER BY salary ASC
FETCH FIRST 1 ROW ONLY;

-- 2) Select department name, employee name, salary for HR or Purchasing departments and rank salaries
SELECT d.department_name,
       e.first_name || ' ' || e.last_name AS employee_name,
       e.salary,
       RANK() OVER (PARTITION BY d.department_name ORDER BY e.salary DESC) AS salary_rank
FROM employees e
JOIN departments d ON e.department_id = d.department_id
WHERE d.department_name IN ('Human Resources', 'Purchasing');

-- 3) Select the 3 employees with minimum salary for department id 50
SELECT *
FROM employees
WHERE department_id = 50
ORDER BY salary ASC
FETCH FIRST 3 ROWS ONLY;

-- 4) Show first name, last name, salary and previous employee’s salary for IT_PROG ordered by hire_date
SELECT first_name, last_name, salary,
       LAG(salary) OVER (ORDER BY hire_date) AS prev_salary
FROM employees
WHERE job_id = 'IT_PROG'
ORDER BY hire_date;

-- 5) Display current job details for employees who worked as IT Programmers in the past
SELECT DISTINCT e.*
FROM employees e
JOIN job_history jh ON e.employee_id = jh.employee_id
WHERE jh.job_id = 'IT_PROG';

-- 6) Make a copy of employees table and update salaries with max salary in their departments
CREATE TABLE employees_copy AS SELECT * FROM employees;

UPDATE employees_copy ec
SET ec.salary = (
  SELECT MAX(salary)
  FROM employees e
  WHERE e.department_id = ec.department_id
);

-- 7) Make a copy of employees table and update salaries with 30% increase
CREATE TABLE employees_copy2 AS SELECT * FROM employees;

UPDATE employees_copy2
SET salary = salary * 1.3;