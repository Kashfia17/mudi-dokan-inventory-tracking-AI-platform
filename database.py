
import sqlite3
import csv
import io
import hashlib
import random

from pathlib import Path
from datetime import datetime, timedelta


# ==========================================
# DATABASE LOCATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "smartstock.db"


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_connection():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# ==========================================
# INITIALIZE DATABASE
# ==========================================

def initialize_db():

    conn = get_connection()

    try:
        cursor = conn.cursor()

        # Products table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (

                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                cost_price REAL NOT NULL,
                selling_price REAL NOT NULL,
                stock_quantity INTEGER NOT NULL DEFAULT 0,
                reorder_level INTEGER NOT NULL DEFAULT 10,
                created_at TEXT NOT NULL

            )
        """)

        # Sales table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales (

                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price REAL NOT NULL,
                total_amount REAL NOT NULL,
                sold_at TEXT NOT NULL,

                FOREIGN KEY (product_id)
                REFERENCES products(id)
                ON DELETE RESTRICT

            )
        """)

        conn.commit()

    finally:
        conn.close()


# ==========================================
# ADD PRODUCT
# ==========================================

def add_product(
    name,
    category,
    cost_price,
    selling_price,
    stock_quantity,
    reorder_level=10
):

    name = name.strip()
    category = category.strip()

    if not name or not category:
        raise ValueError("Product name and category are required.")

    if cost_price < 0 or selling_price < 0:
        raise ValueError("Prices cannot be negative.")

    if stock_quantity < 0 or reorder_level < 0:
        raise ValueError("Stock values cannot be negative.")

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO products (
                name,
                category,
                cost_price,
                selling_price,
                stock_quantity,
                reorder_level,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            category,
            cost_price,
            selling_price,
            stock_quantity,
            reorder_level,
            datetime.now().isoformat(timespec="seconds")
        ))

        conn.commit()

        return cursor.lastrowid

    finally:
        conn.close()


# ==========================================
# GET ALL PRODUCTS
# ==========================================

def get_products():

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM products
            ORDER BY id DESC
        """)

        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


# ==========================================
# UPDATE STOCK
# ==========================================

def update_stock(product_id, new_quantity):

    if new_quantity < 0:
        raise ValueError("Stock cannot be negative.")

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE products
            SET stock_quantity = ?
            WHERE id = ?
        """, (new_quantity, product_id))

        conn.commit()

        return cursor.rowcount > 0

    finally:
        conn.close()


# ==========================================
# RECORD A SALE
# ==========================================

def record_sale(product_id, quantity):

    if quantity <= 0:
        raise ValueError(
            "Sale quantity must be greater than zero."
        )

    conn = get_connection()

    try:
        cursor = conn.cursor()

        # Get product
        cursor.execute("""
            SELECT *
            FROM products
            WHERE id = ?
        """, (product_id,))

        product = cursor.fetchone()

        if product is None:
            raise ValueError("Product not found.")

        current_stock = product["stock_quantity"]

        # Check available stock
        if quantity > current_stock:
            raise ValueError(
                f"Insufficient stock! Only {current_stock} "
                "units available."
            )

        unit_price = product["selling_price"]
        total_amount = unit_price * quantity

        sold_at = datetime.now().isoformat(
            timespec="seconds"
        )

        # Insert sale
        cursor.execute("""
            INSERT INTO sales (
                product_id,
                quantity,
                unit_price,
                total_amount,
                sold_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            product_id,
            quantity,
            unit_price,
            total_amount,
            sold_at
        ))

        # Reduce stock
        cursor.execute("""
            UPDATE products
            SET stock_quantity = stock_quantity - ?
            WHERE id = ?
        """, (quantity, product_id))

        conn.commit()

        return {
            "success": True,
            "total_amount": total_amount,
            "remaining_stock": current_stock - quantity
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ==========================================
# GET ALL SALES
# ==========================================

def get_sales():

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                sales.id,
                sales.product_id,
                products.name AS product_name,
                sales.quantity,
                sales.unit_price,
                sales.total_amount,
                sales.sold_at

            FROM sales

            JOIN products
            ON sales.product_id = products.id

            ORDER BY sales.sold_at DESC
        """)

        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


# ==========================================
# GET LOW STOCK PRODUCTS
# ==========================================

def get_low_stock_products():

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM products
            WHERE stock_quantity <= reorder_level
            ORDER BY stock_quantity ASC
        """)

        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


# ==========================================
# UPDATE PRODUCT INFORMATION
# ==========================================

def update_product(
    product_id,
    name,
    category,
    cost_price,
    selling_price,
    stock_quantity,
    reorder_level
):

    name = name.strip()
    category = category.strip()

    if not name or not category:
        raise ValueError(
            "Product name and category are required."
        )

    if cost_price < 0 or selling_price < 0:
        raise ValueError("Prices cannot be negative.")

    if stock_quantity < 0 or reorder_level < 0:
        raise ValueError("Stock values cannot be negative.")

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE products
            SET
                name = ?,
                category = ?,
                cost_price = ?,
                selling_price = ?,
                stock_quantity = ?,
                reorder_level = ?
            WHERE id = ?
        """, (
            name,
            category,
            cost_price,
            selling_price,
            stock_quantity,
            reorder_level,
            product_id
        ))

        conn.commit()

        return cursor.rowcount > 0

    finally:
        conn.close()


# ==========================================
# DELETE PRODUCT
# ==========================================

def delete_product(product_id):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM products
            WHERE id = ?
        """, (product_id,))

        deleted = cursor.rowcount > 0

        conn.commit()

        return deleted

    except sqlite3.IntegrityError:

        conn.rollback()

        raise ValueError(
            "This product has previous sales records. "
            "It cannot be deleted because those records "
            "must be preserved."
        )

    finally:
        conn.close()


# ==========================================
# GENERATE DEMO SALES
# ==========================================

def generate_demo_sales(days=30):

    if days <= 0:
        raise ValueError("Days must be greater than zero.")

    conn = get_connection()

    try:
        # Track demo generation
        conn.execute("""
            CREATE TABLE IF NOT EXISTS demo_sales_batches (
                id INTEGER PRIMARY KEY,
                generated_at TEXT,
                days INTEGER,
                records INTEGER
            )
        """)

        existing = conn.execute("""
            SELECT id
            FROM demo_sales_batches
            WHERE id = 1
        """).fetchone()

        if existing:
            return {
                "success": False,
                "message": "Demo sales have already been generated."
            }

        products = conn.execute("""
            SELECT id, name, category, selling_price
            FROM products
        """).fetchall()

        if not products:
            return {
                "success": False,
                "message": "Please add products first."
            }

        rng = random.Random(42)

        records = 0

        today = datetime.now().date()
        start_date = today - timedelta(days=days - 1)

        for product in products:

            category = product["category"].lower()

            if "grocery" in category:
                base_demand = 8

            elif "dairy" in category:
                base_demand = 10

            elif "snack" in category:
                base_demand = 6

            else:
                base_demand = 5

            for day in range(days):

                sale_date = start_date + timedelta(days=day)

                # Some days have no sales
                if rng.random() < 0.15:
                    continue

                quantity = rng.randint(
                    1,
                    base_demand * 2
                )

                sale_time = datetime.combine(
                    sale_date,
                    datetime.min.time()
                ) + timedelta(
                    hours=rng.randint(9, 20),
                    minutes=rng.randint(0, 59)
                )

                unit_price = product["selling_price"]
                total_amount = quantity * unit_price

                conn.execute("""
                    INSERT INTO sales (
                        product_id,
                        quantity,
                        unit_price,
                        total_amount,
                        sold_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    product["id"],
                    quantity,
                    unit_price,
                    total_amount,
                    sale_time.isoformat()
                ))

                records += 1

        # Record generation
        conn.execute("""
            INSERT INTO demo_sales_batches (
                id,
                generated_at,
                days,
                records
            )
            VALUES (1, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            days,
            records
        ))

        conn.commit()

        return {
            "success": True,
            "message": "Demo sales generated successfully!",
            "records": records,
            "days": days,
            "products": len(products)
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ==========================================
# IMPORT HISTORICAL SALES FROM CSV
# ==========================================

def import_historical_sales(file_bytes):

    conn = get_connection()

    try:
        cursor = conn.cursor()

        # Calculate file hash to prevent duplicate imports
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Create import tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS csv_import_batches (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                file_hash TEXT UNIQUE NOT NULL,

                imported_at TEXT NOT NULL,

                records INTEGER NOT NULL

            )
        """)

        # Check duplicate file
        cursor.execute("""
            SELECT id
            FROM csv_import_batches
            WHERE file_hash = ?
        """, (file_hash,))

        if cursor.fetchone():

            return {
                "success": False,
                "message": "This CSV file has already been imported.",
                "records": 0
            }

        # Decode CSV
        try:
            content = file_bytes.decode("utf-8-sig")

        except UnicodeDecodeError:
            return {
                "success": False,
                "message": "Could not read the CSV file. Please save it as UTF-8.",
                "records": 0
            }

        reader = csv.DictReader(io.StringIO(content))

        required_columns = [
            "product_id",
            "quantity",
            "unit_price",
            "sold_at"
        ]

        # Validate column names
        if not reader.fieldnames:

            return {
                "success": False,
                "message": "The CSV file is empty.",
                "records": 0
            }

        reader.fieldnames = [
            field.strip() if field else field
            for field in reader.fieldnames
        ]

        if not all(
            column in reader.fieldnames
            for column in required_columns
        ):

            missing = [
                column for column in required_columns
                if column not in reader.fieldnames
            ]

            return {
                "success": False,
                "message": (
                    "Missing required columns: "
                    + ", ".join(missing)
                ),
                "records": 0
            }

        rows = list(reader)

        if not rows:

            return {
                "success": False,
                "message": "The CSV file contains no sales records.",
                "records": 0
            }

        # Get existing product IDs
        cursor.execute("SELECT id FROM products")

        valid_product_ids = {
            row["id"] for row in cursor.fetchall()
        }

        prepared_sales = []

        # Validate every row before inserting anything
        for index, row in enumerate(rows, start=2):

            try:
                product_id = int(row["product_id"])
                quantity = int(row["quantity"])
                unit_price = float(row["unit_price"])

                sold_at_text = str(row["sold_at"]).strip()

                # Validate date
                parsed_date = datetime.fromisoformat(
                    sold_at_text.replace("Z", "+00:00")
                )

            except (ValueError, TypeError, KeyError):

                return {
                    "success": False,
                    "message": (
                        "Invalid data in CSV row {index}. "
                        "Check product ID, quantity, price, and date."
                    ),
                    "records": 0
                }

            # Check product
            if product_id not in valid_product_ids:

                return {
                    "success": False,
                    "message": (
                        f"Product ID {product_id} does not exist "
                        f"(CSV row {index})."
                    ),
                    "records": 0
                }

            # Check quantity
            if quantity <= 0:

                return {
                    "success": False,
                    "message": (
                        f"Quantity must be greater than zero "
                        f"(CSV row {index})."
                    ),
                    "records": 0
                }

            # Check price
            if unit_price < 0:

                return {
                    "success": False,
                    "message": (
                        f"Unit price cannot be negative "
                        f"(CSV row {index})."
                    ),
                    "records": 0
                }

            total_amount = quantity * unit_price

            prepared_sales.append((
                product_id,
                quantity,
                unit_price,
                total_amount,
                parsed_date.isoformat(sep=" ")
            ))

        if not prepared_sales:

            return {
                "success": False,
                "message": "No valid sales records were found.",
                "records": 0
            }

        # Insert all sales records
        cursor.executemany("""
            INSERT INTO sales (
                product_id,
                quantity,
                unit_price,
                total_amount,
                sold_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, prepared_sales)

        # Save import history
        cursor.execute("""
            INSERT INTO csv_import_batches (
                file_hash,
                imported_at,
                records
            )
            VALUES (?, ?, ?)
        """, (
            file_hash,
            datetime.now().isoformat(timespec="seconds"),
            len(prepared_sales)
        ))

        conn.commit()

        return {
            "success": True,
            "message": "Historical sales imported successfully!",
            "records": len(prepared_sales)
        }

    except Exception as e:

        conn.rollback()

        return {
            "success": False,
            "message": f"Import failed: {str(e)}",
            "records": 0
        }

    finally:
        conn.close()
