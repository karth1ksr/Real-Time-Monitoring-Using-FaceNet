import pymysql
import cryptography

cnct = pymysql.connect(
    user='karthiktest',
    password='Welcome1',
    host='34.100.229.134',
    database='person_det'
)

cursor = cnct.cursor()

queries = [
    "CREATE TABLE pdata(S_no int AUTO_INCREMENT PRIMARY KEY, Name varchar(20), pid int(10), Status varchar(20),  Cropped_img LONGBLOB, Cameraid varchar(20), Update_time TIMESTAMP)",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('Karthik', 2001, 'Nil', 'Nil', NOW())",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('Vikram', 2002, 'Nil', 'Nil', NOW())",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('Sekhar', 2003, 'Nil', 'Nil', NOW())",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('Ravi', 2004, 'Nil', 'Nil', NOW())",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('Revan', 2005, 'Nil', 'Nil', NOW())",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('Krishna', 2006, 'Nil', 'Nil', NOW())"
]
for query in queries:
    cursor.execute(query)
    cnct.commit()

cursor.close()
cnct.close()
