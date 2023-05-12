#Postgress
"""
import os
import psycopg2
from psycopg2 import extras
from config import DATABASE_URL

# Add these lines to disable SSL certificate verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

conn = psycopg2.connect(DATABASE_URL, sslmode='require')

async def create_tables():
    with conn:
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id SERIAL PRIMARY KEY,
                    ip_address VARCHAR(255) UNIQUE NOT NULL
                );
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS todo (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users (user_id),
                    task_description TEXT NOT NULL,
                    due_date TIMESTAMP,
                    priority INTEGER,
                    status TEXT NOT NULL
                );
            ''')

            print("Tables created successfully.")


async def delete_tables():
    with conn:
        with conn.cursor() as cur:
            cur.execute('''
                DROP TABLE IF EXISTS todo;
            ''')

            cur.execute('''
                DROP TABLE IF EXISTS users;
            ''')

            print("Tables deleted successfully.")

async def execute_query(query, params=None):
    with conn:
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(query, params)
            conn.commit()

            query_type = query.lower().split()[0]

            # If the query is a SELECT statement, fetch results
            if query_type == "select":
                return cur.fetchall()
            # If the query is an INSERT, UPDATE, or DELETE statement, return a custom message
            elif query_type in {"insert", "update", "delete"}:
                action = query_type.capitalize()
                return f"{action} operation executed successfully."
            # If the query is not a SELECT, INSERT, UPDATE, or DELETE statement, return an empty list
            else:
                return []

async def get_user_id_by_ip(ip):
    query = "SELECT user_id FROM users WHERE ip_address = %s"
    result = await execute_query(query, (ip,))
    return result[0]["user_id"] if result else None


async def create_user_with_ip(ip):
    query = "INSERT INTO users (ip_address) VALUES (%s) RETURNING user_id"
    result = await execute_query(query, (ip,))
    return result[0]["user_id"]


# Fetch all the taks
async def get_all_tasks():
    query = "SELECT * FROM todo"
    return await execute_query(query)

"""
#SQLITE
import sqlite3
import asyncio

conn = sqlite3.connect(':memory:', check_same_thread=False)

async def create_tables():
    with conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT UNIQUE NOT NULL
            );
        ''')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS todo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users (user_id),
                task_description TEXT NOT NULL,
                due_date TIMESTAMP,
                priority INTEGER,
                status TEXT NOT NULL
            );
        ''')

        print("Tables created successfully.")

async def delete_tables():
    with conn:
        cur = conn.cursor()
        cur.execute('''
            DROP TABLE IF EXISTS todo;
        ''')

        cur.execute('''
            DROP TABLE IF EXISTS users;
        ''')

        print("Tables deleted successfully.")

async def execute_query(query, params=None):
    with conn:
        cur = conn.cursor()
        cur.execute(query, params or ())
        conn.commit()

        query_type = query.lower().split()[0]

        if query_type == "select":
            return cur.fetchall()
        elif query_type in {"insert", "update", "delete"}:
            action = query_type.capitalize()
            return f"{action} operation executed successfully."
        else:
            return []

async def get_user_id_by_ip(ip):
    query = "SELECT user_id FROM users WHERE ip_address = ?"
    result = await execute_query(query, (ip,))
    return result[0][0] if result else None

async def create_user_with_ip(ip):
    query = "INSERT INTO users (ip_address) VALUES (?)"
    await execute_query(query, (ip,))
    return await get_user_id_by_ip(ip)

async def get_all_tasks():
    query = "SELECT * FROM todo"
    return await execute_query(query)