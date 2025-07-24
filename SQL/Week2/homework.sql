-- 1) Employees who joined in the month of May
SELECT *
FROM employees
WHERE TO_CHAR(hire_date, 'MM') = '05';

-- 2) Employees who joined in the current year
SELECT *
FROM employees
WHERE EXTRACT(YEAR FROM hire_date) = EXTRACT(YEAR FROM SYSDATE);

-- 3) Number of days between system date and 1st January 2011
SELECT SYSDATE - TO_DATE('2011-01-01', 'YYYY-MM-DD') AS days_difference
FROM dual;

-- 4) Maximum salary of employees
SELECT MAX(salary) AS max_salary
FROM employees;

-- 5) Number of employees in each department
SELECT department_id, COUNT(*) AS num_employees
FROM employees
GROUP BY department_id;

-- 6) Employees who joined after 15th of the month
SELECT *
FROM employees
WHERE TO_NUMBER(TO_CHAR(hire_date, 'DD')) > 15;

-- 7) Average salary of employees in each department who have commission percentage
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
WHERE commission_pct IS NOT NULL
GROUP BY department_id;

-- 8) Job ID for jobs with average salary more than 10000
SELECT job_id
FROM employees
GROUP BY job_id
HAVING AVG(salary) > 10000;

-- 9) Job ID, number of employees, sum of salary, and salary range (max-min)
SELECT 
    job_id,
    COUNT(*) AS num_employees,
    SUM(salary) AS total_salary,
    MAX(salary) - MIN(salary) AS salary_range
FROM employees
GROUP BY job_id;

-- 10) Manager ID and number of employees managed
SELECT manager_id, COUNT(*) AS num_employees
FROM employees
WHERE manager_id IS NOT NULL
GROUP BY manager_id;