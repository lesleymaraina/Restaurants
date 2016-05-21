import sqlite3
import csv

DBPATH = '/Users/jbbinder/Desktop/sql_rdb.db'
conn = sqlite3.connect(DBPATH)
cur = conn.cursor()

query = ("select name, street, city, state, zip, latitude, longitude, "
"cuisinetype, cuisinetype_2, avg(price) as avg_price, max(price) as max_price, "
"case when max(price) > 25 then 'High' when max(price) >= 10 then 'Medium' else 'Low' end as price_type, "
"sum(1) as num_menu_items, avg_rating "
"from r where price != '' "
"group by name, street, city, state, zip, latitude, longitude, cuisinetype, cuisinetype_2, avg_rating")

print(query)


results = cur.execute(query)

with open('/Users/jbbinder/Desktop/query_output.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'street', 'city', 'state', 'zip', 'latitude', 'longitude', 'cuisinetype', 'cuisinetype_2', 'avg_price', 'max_price', 'price_type', 'num_menu_items', 'avg_rating'])
    writer.writerows(results)
