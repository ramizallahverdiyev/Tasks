-- 1. Display full name, salary, commission_pct, hire_date where salary < 10000
SELECT 
    first_name || ' ' || last_name AS full_name,
    salary,
    commission_pct,
    hire_date
FROM employees
WHERE salary < 10000;


-- 2. Display city names (unique) in ascending order
SELECT DISTINCT city
FROM locations
ORDER BY city ASC;


-- 3. Employees who are IT Programmer or Sales Manager and hired between 2002 and 2005
SELECT 
    first_name,
    hire_date,
    job_id
FROM employees
WHERE job_id IN ('IT_PROG', 'SA_MAN')
  AND hire_date BETWEEN TO_DATE('2002-01-01', 'YYYY-MM-DD') AND TO_DATE('2005-12-31', 'YYYY-MM-DD');


-- 4. Display jobs in descending order of job_title
SELECT *
FROM jobs
ORDER BY job_title DESC;


-- 5. Employees where commission_pct is null, salary between 5000-10000, department_id = 30
SELECT *
FROM employees
WHERE commission_pct IS NULL
  AND salary BETWEEN 5000 AND 10000
  AND department_id = 30;


-- 6. Employees who joined after 1st January 2008
SELECT *
FROM employees
WHERE hire_date > TO_DATE('2008-01-01', 'YYYY-MM-DD');


-- 7. Details of employees with ID 150, 160, or 170
SELECT *
FROM employees
WHERE employee_id IN (150, 160, 170);


-- 8. Employees where first_name or last_name starts with S
SELECT *
FROM employees
WHERE first_name LIKE 'S%' OR last_name LIKE 'S%';


-- 9. Length of first name where last name has 'b' after 3rd position
SELECT 
    first_name,
    LENGTH(first_name) AS name_length,
    last_name
FROM employees
WHERE INSTR(last_name, 'b', 4) > 0;
