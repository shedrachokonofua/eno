#!/usr/bin/env python3
"""Simple duckdb query runner."""
import sys
import duckdb

if len(sys.argv) < 2:
    print("Usage: duckdb_query.py 'SQL QUERY'")
    sys.exit(1)

query = sys.argv[1]
con = duckdb.connect()
result = con.execute(query).fetchdf()
print(result.to_string())

