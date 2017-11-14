# This file reads a csv file and creates an external table in Hive and populates it with data so we can execute SQL queries
# and run models only on selected data
# Steps for Ubuntu
# 1) Download Hive from http://www-eu.apache.org/dist/hive/hive-2.3.1/apache-hive-2.3.1-bin.tar.gz and extract
# 2) Downalod Hadoop from http://www-us.apache.org/dist/hadoop/common/hadoop-2.8.2/hadoop-2.8.2.tar.gz and extract
# 3) Set the following environment variables
#    $ export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/"
#    $ export PATH="$PATH:$JAVA_HOME/bin"
#    $ export HADOOP_HOME=/home/aroot/hadoop-2.8.2
#    $ export PATH=$PATH:$HADOOP_HOME/bin
#    $ export PATH=$PATH:$HADOOP_HOME/sbin
#    $ export HIVE_HOME=/home/aroot/apache-hive-2.3.1-bin
#    $ export PATH=$PATH:$HIVE_HOME/bin
# 4) Install MySQL to store metadata (you can use derby also)
#    a) $ sudo apt-get install mysql-server mysql-client
#    b) Install $ mysql_secure_installation
#    c) Check if MySQL server is running : $ systemctl status mysql.service  
#    d) JDBC driver needed for connecting to MySQL : $ sudo apt-get install libmysql-java
#       $ export CLASSPATH=$CLASSPATH:/usr/share/java/mysql-connector-java.jar
#       $ cd $HIVE_HOME/lib
#       $ ln -s /usr/share/java/mysql.jar mysql.jar
#    e) Test if MySQL server is running using : $ mysql -u root --password=password --execute="show databases;"
# 5) Create hive-site.xml which contains MySQL connection details
# cat > $HIVE_HOME/conf/hive-site.xml
#<property>
#  <name>hive.server2.thrift.port</name>
#  <value>10000</value>
#  <description>Port number of HiveServer2 Thrift interface. 
#  Can be overridden by setting $HIVE_SERVER2_THRIFT_PORT</description>
#</property>
#<property>
#  <name>hive.server2.thrift.bind.host</name>
#  <value>127.0.0.1</value>
#  <description>Bind host on which to run the HiveServer2 Thrift interface. 
#  Can be overridden by setting $HIVE_SERVER2_THRIFT_BIND_HOST</description>
#</property>
#<property>
#  <name>hive.server2.authentication</name>
#  <value>NONE</value>
#  <description>Client authentication NONE</description>
#</property>
#<configuration>
#  <property>
#    <name>hive.metastore.warehouse.dir</name>
#    <value>/tmp/hive/warehouse</value>
#    <description>location of default database for the warehouse</description>
#  </property>
#  <property>
#    <name>javax.jdo.option.ConnectionURL</name>
#    <value>jdbc:mysql://127.0.0.1:3306/metastore?createDatabaseIfNotExist=true</value>
#    <description>JDBC connect string for a JDBC metastore</description>
#  </property>
#  <property>
#    <name>javax.jdo.option.ConnectionDriverName</name>
#    <value>com.mysql.jdbc.Driver</value>
#    <description>Driver class name for a JDBC metastore</description>
#  </property>
#  <property>
#    <name>javax.jdo.option.ConnectionUserName</name>
#    <value>root</value>
#    <description>username to use against metastore database</description>
#  </property>
#  <property>
#    <name>javax.jdo.option.ConnectionPassword</name>
#    <value>password</value>
#    <description>password to use against metastore database</description>
#  </property>
#</configuration>
# 6) In Hadoop run start-all script $HADOOP_HOME/sbin/start-all.sh and then run the follwoing commands
#    $ hadoop fs -mkdir       /tmp/
#    $ hadoop fs -mkdir       /tmp/hive
#    $ hadoop fs -mkdir       /tmp/hive/warehouse
#    $ hadoop fs -chmod g+w   /tmp/hive
#    $ hadoop fs -chmod g+w   /tmp/hive/warehouse
# 7) Set MYSQL as Database type using schematool
#    $HIVE_HOME/bin/schematool -dbType mysql -initSchema --verbose
# 8) Start Metastore service 
#    $ hive --service metastore
# 9) Start HiveServer2 server
#    $ hive --service hiveserver2 
# 10) Test using hive client beeline
#   $HIVE_HOME/bin/beeline -u jdbc:hive2://localhost:10000
#     beeline> CREATE SCHEMA IF NOT EXISTS myschema;
#     beeline> CREATE EXTERNAL TABLE IF NOT EXISTS myschema.mytable(ID String, LIMIT_BAL INT, SEX String, EDUCATION String, \
#     MARRIAGE String, AGE INT, PAY_0 INT, PAY_2 INT, PAY_3 INT, PAY_4 INT, PAY_5 INT, PAY_6 INT, \
#     BILL_AMT1 INT, BILL_AMT2 INT, BILL_AMT3 INT, BILL_AMT4 INT, BILL_AMT5 INT, BILL_AMT6 INT, \
#     PAY_AMT1 INT, PAY_AMT2 INT, PAY_AMT3 INT, PAY_AMT4 INT, PAY_AMT5 INT, PAY_AMT6 INT, y INT) \
#     COMMENT 'Data about credit card fraud from a public database' \
#     ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE
#     beeline> LOAD DATA LOCAL INPATH '/home/aroot/UCI_Credit_Card.csv' OVERWRITE INTO TABLE myschema.mytable;
#     beeline> select * from myschema.mytable;       
# 11) Install pyhive from https://pypi.python.org/pypi/PyHive click on Download PyHive-0.5.0.tar.gz i.e. from the link
# https://pypi.python.org/packages/1f/7e/507119f318a317078d36bdc308d31dec3a42d5edbeff69e7c3cc3ca865bd/PyHive-0.5.0.tar.gz
#   $ cd PyHive-0.5.0
#   $ python3 ./setup.py build
#   $ python3 ./setup.py install

from pyhive import hive # refer https://pypi.python.org/pypi/PyHive for usage
import pandas as pd

# If hiveserver2 is running on local host port 10000
conn = hive.Connection(host="127.0.0.1", port=10000)
cursor = conn.cursor()

# cleanup older tables if any
cmd="DROP TABLE IF EXISTS myschema1.mytable"
cursor.execute(cmd)
cmd0="DROP DATABASE IF EXISTS myschema1"
cursor.execute(cmd0)

# SCHEMA and DATABASE can be used interchangably
cmd1="CREATE SCHEMA IF NOT EXISTS myschema1"
cursor.execute(cmd1)

# I have manually changed default.payment.next.month to y in the csv file otherwise the following command was failing
cmd2="CREATE EXTERNAL TABLE IF NOT EXISTS myschema1.mytable (ID String, \
LIMIT_BAL INT, SEX String, EDUCATION String, MARRIAGE String, AGE INT, \
PAY_0 INT, PAY_2 INT, PAY_3 INT, PAY_4 INT, PAY_5 INT, PAY_6 INT, \
BILL_AMT1 INT, BILL_AMT2 INT, BILL_AMT3 INT, BILL_AMT4 INT, BILL_AMT5 INT, BILL_AMT6 INT, \
PAY_AMT1 INT, PAY_AMT2 INT, PAY_AMT3 INT, PAY_AMT4 INT, PAY_AMT5 INT, PAY_AMT6 INT, y INT) \
ROW FORMAT DELIMITED FIELDS TERMINATED BY \',\' STORED AS TEXTFILE"
cursor.execute(cmd2)

# load csv file into the table
cmd3="LOAD DATA LOCAL INPATH '/home/aroot/UCI_Credit_Card.csv' OVERWRITE INTO TABLE myschema1.mytable"
cursor.execute(cmd3)

# we can also convert this into ORC format for faster results

# Now we can run SQL commands to filter the data to get only certain customers
cmd4="SELECT * FROM myschema1.mytable WHERE PAY_AMT4 > 500000"
df = pd.read_sql(cmd4, conn)
print(df.head(5))

# if you do not want to create table in python code, etc. the following small piece of code also works
#conn = hive.Connection(host="127.0.0.1", port=10000,  configuration={'hive.exec.reducers.max': '123'})
#cmd="SELECT * FROM myschema1.mytable WHERE PAY_AMT4 > 500000"
#df = pd.read_sql(cmd, conn)
#print(df)
