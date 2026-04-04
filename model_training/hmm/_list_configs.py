#!/usr/bin/env python
import psycopg2, psycopg2.extras, json, sys
conn = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', password=sys.argv[1], dbname='aistock')
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
cur.execute("SELECT config_id, display_name, config_json, created_at FROM model_train_configs WHERE model_type='sector_hmm' ORDER BY created_at DESC")
for r in cur.fetchall():
    cj = r['config_json'] if isinstance(r['config_json'], dict) else json.loads(r['config_json'])
    desc = cj.get('description', '-')
    print(f"{r['config_id'][:8]}  {r['display_name']:<42} desc={desc}")
cur.close(); conn.close()
