-- 1) Job title, difference between min and max salaries where max_salary is between 12000 and 18000
SELECT 
    job_title,
    (max_salary - min_salary) AS salary_difference
FROM jobs
WHERE max_salary BETWEEN 12000 AND 18000;

-- 2) Employees with no commission_pct, salary between 7000 and 12000, not in departments 50, 30, 80
SELECT *
FROM employees
WHERE commission_pct IS NULL
  AND salary BETWEEN 7000 AND 12000
  AND department_id NOT IN (50, 30, 80);

-- 3) Full name, hire date, commission_pct, email-phone, salary where salary > 11000 ordered by full name desc
SELECT 
    first_name || ' ' || last_name AS full_name,
    hire_date,
    commission_pct,
    email || '-' || phone_number AS contact_info,
    salary
FROM employees
WHERE salary > 11000
ORDER BY full_name DESC;

-- 4) Employees with first name ending in 'm' and hired before 5 June 2010
SELECT 
    first_name,
    last_name,
    salary
FROM employees
WHERE first_name LIKE '%m'
  AND hire_date < TO_DATE('2010-06-05', 'YYYY-MM-DD');

-- 5) Full name, contact info, salary where salary NOT between 9000-17000 and commission_pct IS NOT NULL
SELECT 
    first_name || ' ' || last_name AS Full_Name,
    phone_number || '-' || email AS Contact_Details,
    salary AS Remuneration
FROM employees
WHERE salary NOT BETWEEN 9000 AND 17000
  AND commission_pct IS NOT NULL;

-- 6) All info about the Marketing department
SELECT *
FROM departments
WHERE department_name = 'Marketing';

-- 7) job_history data ordered by employee_id DESC, start_date ASC
SELECT *
FROM job_history
ORDER BY employee_id DESC, start_date ASC;

-- 8) job_id, salary where phone starts with 515 or 590 and hired after 2003, ordered by hire_date and salary
SELECT job_id, salary
FROM employees
WHERE (phone_number LIKE '515%' OR phone_number LIKE '590%')
  AND hire_date > TO_DATE('2003-12-31', 'YYYY-MM-DD')
ORDER BY hire_date ASC, salary ASC;

-- 9) Employees hired in 2001
SELECT *
FROM employees
WHERE EXTRACT(YEAR FROM hire_date) = 2001;

-- 10) Employees not hired in 2006 and 2007
SELECT first_name, last_name
FROM employees
WHERE EXTRACT(YEAR FROM hire_date) NOT IN (2006, 2007);

-- 11) Email, job_id, first name of employees hired in 2007 or month is January
SELECT email, job_id, first_name
FROM employees
WHERE EXTRACT(YEAR FROM hire_date) = 2007
   OR EXTRACT(MONTH FROM hire_date) = 1;

-- 12) Employees hired after 2007 or salary < 10000
SELECT *
FROM employees
WHERE EXTRACT(YEAR FROM hire_date) > 2007
   OR salary < 10000;
