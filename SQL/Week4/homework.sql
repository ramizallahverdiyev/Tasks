-- 1. Create STUDENTS, ACTIVITIES, SCHEDULE tables

CREATE TABLE STUDENTS (
    S_ID NUMBER PRIMARY KEY NOT NULL,
    FIRST_NAME VARCHAR2(50),
    LAST_NAME VARCHAR2(50),
    PHONE_NUMBER VARCHAR2(20),
    EMAIL VARCHAR2(100)
);

CREATE TABLE ACTIVITIES (
    A_ID NUMBER PRIMARY KEY NOT NULL,
    A_NAME VARCHAR2(100) NOT NULL,
    COST NUMBER NOT NULL
);

CREATE TABLE SCHEDULE (
    S_ID NUMBER,
    A_ID NUMBER,
    S_DATE DATE,
    FOREIGN KEY (S_ID) REFERENCES STUDENTS(S_ID),
    FOREIGN KEY (A_ID) REFERENCES ACTIVITIES(A_ID)
);

-- 2. Insert data into students table from employees table

INSERT INTO STUDENTS (S_ID, FIRST_NAME, LAST_NAME, PHONE_NUMBER, EMAIL)
SELECT employee_id, first_name, last_name, phone_number, email
FROM employees;

-- 3. Change phone number to ‘***’ for students with s_id > 200

UPDATE STUDENTS
SET PHONE_NUMBER = '***'
WHERE S_ID > 200;

-- 4. Update first name and last names of students in upper cases

UPDATE STUDENTS
SET FIRST_NAME = UPPER(FIRST_NAME),
    LAST_NAME = UPPER(LAST_NAME);

-- 5. Update email to 'DSA' for students with s_id > 150

UPDATE STUDENTS
SET EMAIL = 'DSA'
WHERE S_ID > 150;

-- 6. Create PROGRAMMERS table using records from EMPLOYEES where job_id contains ‘PROG’

CREATE TABLE PROGRAMMERS AS
SELECT *
FROM EMPLOYEES
WHERE JOB_ID LIKE '%PROG%';

-- 7. Delete records from students table where s_id is between 150 and 160

DELETE FROM STUDENTS
WHERE S_ID BETWEEN 150 AND 160;

-- 8a. Insert data into SCHEDULE, then truncate and see results

INSERT INTO SCHEDULE (S_ID, A_ID, S_DATE) VALUES (101, 1, SYSDATE);
INSERT INTO SCHEDULE (S_ID, A_ID, S_DATE) VALUES (102, 2, SYSDATE + 1);
SELECT * FROM SCHEDULE;

TRUNCATE TABLE SCHEDULE;

-- 8b. Drop SCHEDULE table

DROP TABLE SCHEDULE;

-- 9. Date calculations (use a specific date or SYSDATE)

-- a) First and last day of the next year
SELECT
  ADD_MONTHS(TRUNC(SYSDATE, 'YEAR'), 12) AS first_day_next_year,
  ADD_MONTHS(TRUNC(SYSDATE, 'YEAR'), 24) - 1 AS last_day_next_year
FROM dual;

-- b) First and last day of the next month
SELECT
  TRUNC(ADD_MONTHS(SYSDATE, 1), 'MM') AS first_day_next_month,
  LAST_DAY(ADD_MONTHS(SYSDATE, 1)) AS last_day_next_month
FROM dual;

-- c) First and last day of the previous month
SELECT
  TRUNC(ADD_MONTHS(SYSDATE, -1), 'MM') AS first_day_prev_month,
  LAST_DAY(ADD_MONTHS(SYSDATE, -1)) AS last_day_prev_month
FROM dual;

-- 10. Create table “Participants” from EMPLOYEES where salary > 10000

CREATE TABLE PARTICIPANTS AS
SELECT FIRST_NAME, LAST_NAME, SALARY
FROM EMPLOYEES
WHERE SALARY > 10000;