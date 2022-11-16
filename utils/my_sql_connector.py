import mysql.connector
import platform


def get_mydb():
    if 'Windows' in platform.system():
        mydb_win = mysql.connector.connect(
            host="localhost",
            user="your_user",
            password="your_password",
            database="your_database"
        )
        return mydb_win
    else:
        mydb_lin = mysql.connector.connect(
            host="123.456.789.10",
            user="your_user",
            password="your_password",
            database="your_database",
            auth_plugin='mysql_native_password'
        )
        return mydb_lin


mydb = get_mydb()
mycursor = mydb.cursor()


def get_table_names(symbol):
    mycursor.execute("show tables;")
    myresult = mycursor.fetchall()

    oot = []

    for x in myresult:
        tmp1 = str(x)
        tmp2 = tmp1.replace("('", '')
        tmp3 = tmp2.replace("',)", '')

        if symbol in tmp3:
            oot.append(tmp3)

    return oot


def get_number_of_rows_in_a_table(table_name):
    mycursor.execute("SELECT COUNT(*) FROM " + table_name + ";")
    myresult = mycursor.fetchall()

    tmp1 = str(myresult)
    tmp2 = tmp1.replace("[(", "")
    tmp3 = tmp2.replace(",)]", "")
    tmp4 = int(tmp3)

    return tmp4


def get_all_rows_from_table(table_name):
    mydb.ping(reconnect=True)

    mycursor.execute("SELECT * FROM " + table_name + ";")
    myresult = mycursor.fetchall()

    oot = []

    for x in myresult:
        oot.append(x)

    return oot


def _get_specific_rows_from_table_old(table_name, start_index, end_index):
    mycursor.execute("SELECT * FROM " + table_name + ";")
    myresult = mycursor.fetchall()

    oot = []

    for x in myresult:
        oot.append(x)

    return oot[start_index: end_index - 1]


def get_specific_rows_from_table(table_name, start_index, end_index):
    offset = str(start_index)
    num_rows = str(end_index - start_index - 1)

    mycursor.execute("SELECT * FROM " + table_name + " LIMIT " + offset + ", " + num_rows + ";")
    myresult = mycursor.fetchall()

    oot = []

    for x in myresult:
        oot.append(x)

    return oot
