#!/usr/bin/env python
import psycopg2, psycopg2.extras, json, sys
conn = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', password=sys.argv[1], dbname='aistock')
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
cur.execute("""
    SELECT c.display_name, s.snapshot_id, s.metrics_json
    FROM model_train_configs c
    JOIN model_train_snapshots s ON s.config_id = c.config_id
    WHERE c.model_type='sector_hmm' AND s.status='completed'
    ORDER BY s.trained_at DESC LIMIT 3
""")
for r in cur.fetchall():
    mj = r['metrics_json']
    keys = list(mj.keys()) if isinstance(mj, dict) else list(json.loads(mj).keys()) if mj else []
    print(f"{r['display_name'][:40]}  snapshot={r['snapshot_id'][:8]}  metrics_keys={keys[:10]}")
cur.close(); conn.close()
