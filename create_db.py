#import necessary modules
import pymysql
import cryptography

cnct = pymysql.connect(
    user='enter_username',
    password='enter_password',
    host='enter_host_name',
    database='your_db_name'
)

cursor = cnct.cursor()

queries = [
    "CREATE TABLE pdata(S_no int AUTO_INCREMENT PRIMARY KEY, Name varchar(20), pid int(10), Status varchar(20),  Cropped_img LONGBLOB, Cameraid varchar(20), Update_time TIMESTAMP)",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('insert the values')",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('insert the values')",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('insert the values')",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('insert the values')",
    "INSERT INTO pdata(Name, pid, Status, Cameraid, Update_time) values('insert the values')",
]
for query in queries:
    cursor.execute(query)
    cnct.commit()

cursor.close()
cnct.close()

