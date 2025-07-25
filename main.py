import duckdb
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import time
import numpy as np
import pandas as pd

app = FastAPI()


app.add_middleware(
    CORSMiddleware, # type: ignore
    allow_origins=["http://localhost:3000", "https://shopify-data.netlify.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "app/shopify_data.duckdb"
TABLE_NAME = "shopify_data"
CSV_PATH = "shopify_data.csv"

logging.basicConfig(level=logging.INFO)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Shopify DuckDB Search API!"}

@app.get("/search")
def search_duckdb(q: str = Query(""), column: Optional[str] = Query(None)):
    start = time.time()
    try:
        conn = duckdb.connect(DB_PATH)
        
        if column:
            where_clause = f"LOWER({column}) LIKE '%{q.lower().replace("'", "''")}%'" if q else "TRUE"
        else:
            # Get column names + types from DuckDB
            describe = conn.execute(f"DESCRIBE {TABLE_NAME}").fetchall()
            columns_list = [(row[0], row[1]) for row in describe]

            if q:
                q_escaped = q.lower().replace("'", "''")
                or_clauses = []
                for col, col_type in columns_list:
                    if "CHAR" in col_type or "TEXT" in col_type or "STRING" in col_type or "VARCHAR" in col_type:
                        or_clauses.append(f"LOWER({col}) LIKE '%{q_escaped}%'")
                    else:
                        or_clauses.append(f"CAST({col} AS VARCHAR) LIKE '%{q_escaped}%'")
                where_clause = " OR ".join(or_clauses)
            else:
                where_clause = "TRUE"

        query = f"""
            SELECT * 
            FROM {TABLE_NAME}
            WHERE {where_clause}
            LIMIT 100
        """
        result = conn.execute(query).fetchdf()
        columns = result.columns.tolist()
        result = result.replace([np.nan, np.inf, -np.inf], None)
        data = result.to_dict(orient="records")
        conn.close()
    except Exception as e:
        logging.error(f"Error: {e}")
        try:
            conn = duckdb.connect(DB_PATH)
            columns = conn.execute(f"DESCRIBE {TABLE_NAME}").fetchdf()['column_name'].tolist()
            conn.close()
        except Exception:
            columns = []
        return {
            "results": [],
            "columns": columns,
            "error": str(e)
        }

    elapsed = int((time.time() - start) * 1000)
    logging.info(f"/search q='{q}' col='{column}' -> {len(data)} rows in {elapsed}ms")
    return {
        "results": data,
        "columns": columns
    }

@app.post("/update-cell")
async def update_cell(request: Request):
    try:
        data = await request.json()
        row_index = data.get("row_index")
        column = data.get("column")
        value = data.get("value")
        if row_index is None or column is None:
            return JSONResponse({"success": False, "error": "Missing row_index or column"}, status_code=400)
        conn = duckdb.connect(DB_PATH)
        # Get the domain for the row_index
        df = conn.execute(f"SELECT domain FROM {TABLE_NAME} LIMIT 1 OFFSET {row_index}").fetchdf()
        if df.empty:
            conn.close()
            return JSONResponse({"success": False, "error": "Invalid row_index"}, status_code=400)
        domain = df.iloc[0]['domain']
        # Update the cell in DuckDB
        conn.execute(f"UPDATE {TABLE_NAME} SET {column} = ? WHERE domain = ?", [value, domain])
        # Get the updated value
        updated = conn.execute(f"SELECT {column} FROM {TABLE_NAME} WHERE domain = ?", [domain]).fetchone()
        cell_value = updated[0] if updated else "Not Available"
        if cell_value in [None, "", "nan", "NaN", "None"]:
            cell_value = "Not Available"
        conn.close()
        return {"success": True, "row_index": row_index, "column": column, "value": cell_value}
    except Exception as e:
        logging.error(f"/update-cell error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)