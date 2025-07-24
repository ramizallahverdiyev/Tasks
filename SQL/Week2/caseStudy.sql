-- 1) Show minimum, average and maximum salary in last 15 years according to job id
SELECT job_id,
       MIN(salary) AS min_salary,
       ROUND(AVG(salary), 2) AS avg_salary,
       MAX(salary) AS max_salary
FROM employees
WHERE hire_date >= ADD_MONTHS(SYSDATE, -12*15)
GROUP BY job_id;

-- 2) How many employees hired after 2005 for each department?
SELECT department_id, COUNT(*) AS num_employees
FROM employees
WHERE EXTRACT(YEAR FROM hire_date) > 2005
GROUP BY department_id;

-- 3) Departments in which the difference between maximum and minimum salary is greater than 5000
SELECT department_id
FROM employees
GROUP BY department_id
HAVING MAX(salary) - MIN(salary) > 5000;

-- 4) Display salaries of employees who has no commission pact according to departments (without using WHERE)
SELECT department_id, salary
FROM employees
MODEL RETURN UPDATED ROWS
PARTITION BY (department_id)
DIMENSION BY (ROWNUM AS rn)
MEASURES (commission_pct, salary)
RULES UPSERT (
  salary[ANY] = CASE WHEN commission_pct[CV()] IS NULL THEN salary[CV()] END
);

-- Alternatif (daha basit ve anlaşılır):
SELECT department_id, salary
FROM employees
GROUP BY department_id, salary, commission_pct
HAVING commission_pct IS NULL;

-- 5) How many people has job id with average salary between 3000 and 7000?
SELECT job_id, COUNT(*) AS num_employees
FROM employees
GROUP BY job_id
HAVING AVG(salary) BETWEEN 3000 AND 7000;

-- 6) Find number of employees with same name
SELECT first_name, COUNT(*) AS occurrences
FROM employees
GROUP BY first_name
HAVING COUNT(*) > 1;

-- 7) How many people with the same phone code work in departments 50 and 90?
SELECT SUBSTR(phone_number, 1, 3) AS phone_code, COUNT(*) AS num_employees
FROM employees
WHERE department_id IN (50, 90)
GROUP BY SUBSTR(phone_number, 1, 3)
HAVING COUNT(*) > 1;

-- 8) Departments with avg number of employees > 5 in spring and autumn (Mar-May, Sep-Nov)
SELECT department_id, COUNT(*) AS num_employees
FROM employees
WHERE TO_CHAR(hire_date, 'MM') IN ('03', '04', '05', '09', '10', '11')
GROUP BY department_id
HAVING COUNT(*) / 2 > 5;

-- 9) How many employees work in departments which has max salary more than 5000?
SELECT COUNT(*) AS num_employees
FROM employees e
WHERE e.department_id IN (
    SELECT department_id
    FROM employees
    GROUP BY department_id
    HAVING MAX(salary) > 5000
);

-- 10) Change second letter of employees’ names with the last letter and display
SELECT first_name,
       SUBSTR(first_name, 1, 1) ||
       SUBSTR(first_name, -1, 1) ||
       SUBSTR(first_name, 3) AS modified_name
FROM employees
WHERE LENGTH(first_name) >= 2;
