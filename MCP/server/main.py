# from fastmcp import FastMCP

# mcp = FastMCP(name="DEMO_SERVER")
# # this is a mcp server instance


# @mcp.tool
# def add_2_numbers(a: float, b: float) -> float:
#     return a + b


# # an mcp tool to add 2 numbers.

# # uv run fastmcp dev filename
# # for debugging


# if __name__ == "__main__":
#     mcp.run()


import os
import sqlite3
from sqlite3.dbapi2 import sqlite_version

from fastmcp import FastMCP

mcp = FastMCP(name="ExpenseTracker")


DB_PATH = os.path.join(os.path.dirname(__file__), "expenses.db")
print(DB_PATH)


def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS expenses(
            id INTEGER PRIMARY KEY AUTOINCREMENT.
            date TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT DEFAULT '',
            note TEXT DEFAULT ''
            )
            """)


init_db()


@mcp.tool()
def add_expenses(date, amount, category, subcategory="", note=""):
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            "INSERT INTO expenses(date , amount , category , subcategory , note) VALUES (?,?,?,?,?)",
            (date, amount, category, subcategory, note),
        )
        return {"status": "ok", "id": cur.lastrowid}


@mcp.tool()
def list_expenses():
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            "SELECT id , date , amount , category , subcategory , note FROM expenses ORDER BY id ASX"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@mcp.resource("expense://categories", mime_type="application/json")
def categories():
    with open("a category json file", "r", encoding="utf-8") as f:
        return f.read()


# if u want to send the client a resource , lets say an image or a file to refer to , we can do this via mcp.resource.

# first build a fastAPI server then can easily convert the server to a fastmcp.
if __name__ == "__main__":
    mcp.run()
