import psycopg2, sys
conn = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', password=sys.argv[1], dbname='aistock')
cur = conn.cursor()
cur.execute("SELECT strategy_id, display_name, source_code_relpath, length(source_code) as code_len FROM aistock_strategy_catalog")
for r in cur.fetchall():
    print(f"  {r[0]:<25} {r[1]:<40} path={r[2]}  code_len={r[3]}")
cur.close(); conn.close()
