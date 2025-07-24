-- 1. Display the first promotion year for each employee.
SELECT employee_id,
       MIN(EXTRACT(YEAR FROM start_date)) AS first_promotion_year
FROM job_history
GROUP BY employee_id;

-- 2. Display location, city and department name of employees who have been promoted more than once.
SELECT l.street_address, l.city, d.department_name
FROM job_history jh
JOIN departments d ON jh.department_id = d.department_id
JOIN locations l ON d.location_id = l.location_id
WHERE jh.employee_id IN (
    SELECT employee_id
    FROM job_history
    GROUP BY employee_id
    HAVING COUNT(*) > 1
);

-- 3. Display minimum and maximum “hire_date” of employees work in IT and HR departments.
SELECT d.department_name,
       MIN(e.hire_date) AS min_hire_date,
       MAX(e.hire_date) AS max_hire_date
FROM employees e
JOIN departments d ON e.department_id = d.department_id
WHERE d.department_name IN ('IT', 'HR')
GROUP BY d.department_name;

-- 4. Find difference between current date and hire dates of employees after sorting them by hire date, then show difference in days, months and years.
SELECT first_name || ' ' || last_name AS employee_name,
       hire_date,
       TRUNC(SYSDATE - hire_date) AS diff_days,
       TRUNC(MONTHS_BETWEEN(SYSDATE, hire_date)) AS diff_months,
       TRUNC(MONTHS_BETWEEN(SYSDATE, hire_date)/12) AS diff_years
FROM employees
ORDER BY hire_date;

-- 5. Find which departments used to hire earliest/latest.
SELECT d.department_name, MIN(e.hire_date) AS first_hired, MAX(e.hire_date) AS last_hired
FROM employees e
JOIN departments d ON e.department_id = d.department_id
GROUP BY d.department_name
ORDER BY MIN(e.hire_date), MAX(e.hire_date) DESC;

-- 6. Find the number of departments with no employee for each city.
SELECT l.city, COUNT(d.department_id) AS dept_without_employees
FROM departments d
JOIN locations l ON d.location_id = l.location_id
LEFT JOIN employees e ON d.department_id = e.department_id
WHERE e.employee_id IS NULL
GROUP BY l.city;

-- 7. Create a category called “seasons” and find in which season most employees were hired.
SELECT season, COUNT(*) AS hires
FROM (
    SELECT CASE
             WHEN EXTRACT(MONTH FROM hire_date) IN (12,1,2) THEN 'Winter'
             WHEN EXTRACT(MONTH FROM hire_date) IN (3,4,5) THEN 'Spring'
             WHEN EXTRACT(MONTH FROM hire_date) IN (6,7,8) THEN 'Summer'
             WHEN EXTRACT(MONTH FROM hire_date) IN (9,10,11) THEN 'Autumn'
           END AS season
    FROM employees
)
GROUP BY season
ORDER BY hires DESC;

-- 8. Find the cities of employees with average salary more than 5000.
SELECT l.city
FROM employees e
JOIN departments d ON e.department_id = d.department_id
JOIN locations l ON d.location_id = l.location_id
GROUP BY l.city
HAVING AVG(e.salary) > 5000;