-- 1. Display last name, job title of employees who have commission percentage and belong to department 30
SELECT last_name, job_title
FROM employees e
JOIN jobs j ON e.job_id = j.job_id
WHERE e.commission_pct IS NOT NULL
  AND e.department_id = 30;

-- 2. Display department name, manager name, and salary of the manager for all managers whose experience is more than 5 years
SELECT d.department_name,
       m.first_name || ' ' || m.last_name AS manager_name,
       m.salary
FROM departments d
JOIN employees m ON d.manager_id = m.employee_id
WHERE MONTHS_BETWEEN(SYSDATE, m.hire_date) > 12 * 5;

-- 3. Display employee name if the employee joined before his manager
SELECT e.first_name || ' ' || e.last_name AS employee_name
FROM employees e
JOIN employees m ON e.manager_id = m.employee_id
WHERE e.hire_date < m.hire_date;

-- 4. Display employee name, job title for the jobs employee did in the past where the job was done less than six months
SELECT e.first_name || ' ' || e.last_name AS employee_name,
       j.job_title
FROM job_history jh
JOIN employees e ON jh.employee_id = e.employee_id
JOIN jobs j ON jh.job_id = j.job_id
WHERE MONTHS_BETWEEN(jh.end_date, jh.start_date) < 6;

-- 5. Display department name, average salary and number of employees with commission within the department
SELECT d.department_name,
       ROUND(AVG(e.salary), 2) AS avg_salary,
       COUNT(e.employee_id) AS num_with_commission
FROM employees e
JOIN departments d ON e.department_id = d.department_id
WHERE e.commission_pct IS NOT NULL
GROUP BY d.department_name;

-- 6. Display employee name and country in which he is working
SELECT e.first_name || ' ' || e.last_name AS employee_name,
       c.country_name
FROM employees e
JOIN departments d ON e.department_id = d.department_id
JOIN locations l ON d.location_id = l.location_id
JOIN countries c ON l.country_id = c.country_id;