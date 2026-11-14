
import pandas as pd
from datetime import datetime, timedelta

from database import (
    get_products,
    get_sales,
    get_low_stock_products
)


def answer_question(question):

    question = question.lower().strip()

    products = get_products()
    sales = get_sales()
    low_stock = get_low_stock_products()

    if not question:
        return "Please enter a question."

    if sales:
        df = pd.DataFrame(sales)
        df["sold_at"] = pd.to_datetime(df["sold_at"])
    else:
        df = pd.DataFrame()

    # 1. Low stock
    if any(word in question for word in [
        "low stock",
    "low-stock",
    "running out of stock",
    "run out of stock",
    "almost out of stock",
    "out of stock",
    "stock running low",
    "products running out"
    ]):

        if not low_stock:
            return "All products currently have sufficient stock."

        result = "Products that need attention:\n\n"

        for p in low_stock:
            result += (
                f"- {p['name']}: {p['stock_quantity']} units "
                f"(Reorder level: {p['reorder_level']})\n"
            )

        return result

    # 2. Best-selling products
    if any(word in question for word in [
        "best selling", "best-selling", "top selling",
        "most popular", "sells the most", "popular product"
    ]):

        if df.empty:
            return "No sales records are available."

        best = (
            df.groupby("product_name")["quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )

        result = "Your top-selling products are:\n\n"

        for name, quantity in best.items():
            result += f"- {name}: {quantity} units sold\n"

        return result

    # 3. Product-specific questions
    for product in products:

        name = product["name"].lower()

        if name in question:

            if any(word in question for word in [
                "stock", "available", "left", "remaining"
            ]):
                return (
                    f"You currently have {product['stock_quantity']} "
                    f"units of {product['name']} in stock."
                )

            if any(word in question for word in [
                "price", "cost", "how much"
            ]):
                return (
                    f"The selling price of {product['name']} "
                    f"is ৳{product['selling_price']:,.2f} per unit."
                )

            if any(word in question for word in [
                "sold", "sales", "sell"
            ]):

                if df.empty:
                    return "No sales records are available."

                product_sales = df[
                    df["product_id"] == product["id"]
                ]

                quantity = int(product_sales["quantity"].sum())

                return (
                    f"You have sold {quantity} units of "
                    f"{product['name']} in total."
                )

    # 4. Yesterday's revenue
    if "yesterday" in question and any(word in question for word in [
        "revenue", "sales", "money", "earn", "make"
    ]):

        if df.empty:
            return "No sales records are available."

        yesterday = (datetime.now() - timedelta(days=1)).date()

        yesterday_sales = df[
            df["sold_at"].dt.date == yesterday
        ]

        revenue = yesterday_sales["total_amount"].sum()

        return f"Yesterday's revenue was ৳{revenue:,.2f}."

    # 5. Today's revenue
    if any(word in question for word in ["today", "today's"]) and any(
        word in question for word in [
            "revenue", "sales", "money", "earn", "make"
        ]
    ):

        if df.empty:
            return "No sales records are available."

        today = datetime.now().date()

        today_sales = df[
            df["sold_at"].dt.date == today
        ]

        revenue = today_sales["total_amount"].sum()

        return f"Today's revenue is ৳{revenue:,.2f}."

    # 6. Monthly revenue
    if any(word in question for word in [
        "month", "monthly"
    ]) and any(word in question for word in [
        "revenue", "sales", "money", "earn", "make"
    ]):

        if df.empty:
            return "No sales records are available."

        now = datetime.now()

        monthly_sales = df[
            (df["sold_at"].dt.month == now.month)
            & (df["sold_at"].dt.year == now.year)
        ]

        revenue = monthly_sales["total_amount"].sum()

        return f"This month's revenue is ৳{revenue:,.2f}."

    # 7. Total revenue
    if any(phrase in question for phrase in [
        "total revenue", "overall revenue",
        "total sales", "total earnings",
        "how much money", "how much did my shop make"
    ]):

        if df.empty:
            return "No sales records are available."

        revenue = df["total_amount"].sum()

        return f"Your total recorded sales revenue is ৳{revenue:,.2f}."

    # 8. Total units sold
    if any(phrase in question for phrase in [
        "units sold", "total quantity sold",
        "how many items sold", "how many units sold"
    ]):

        if df.empty:
            return "No sales records are available."

        units = int(df["quantity"].sum())

        return f"You have sold {units} units in total."

    # 9. Total product count
    if any(phrase in question for phrase in [
        "how many products", "total products"
    ]):
        return f"Your shop currently has {len(products)} products."

    # 10. Total available stock
    if any(phrase in question for phrase in [
        "total stock", "stock available",
        "how much stock", "total inventory"
    ]):
        total_stock = sum(p["stock_quantity"] for p in products)
        return f"Your shop currently has {total_stock} units in stock."

    # 11. General fallback
    return (
        "I couldn't understand that question yet. "
        "Try asking about total revenue, today's sales, "
        "monthly revenue, best-selling products, "
        "low stock, or a specific product's stock or price."
    )
