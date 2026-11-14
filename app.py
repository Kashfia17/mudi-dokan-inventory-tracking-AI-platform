
import streamlit as st
import pandas as pd
import plotly.express as px
from chatbot import answer_question
import hashlib
import base64

from prediction import (
    prepare_sales_data,
    train_demand_models,
    load_demand_models,
    predict_tomorrow,
    get_stock_recommendation
)

from database import (
    initialize_db,
    add_product,
    get_products,
    get_sales,
    get_low_stock_products,
    update_product,
    delete_product,
    record_sale,
    import_historical_sales
)

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="SmartStock AI",
    page_icon="📦",
    layout="wide"
)

initialize_db()

# -----------------------------------
# CUSTOM DESIGN
# -----------------------------------

import base64

# =====================================
# LOAD BACKGROUND IMAGES
# =====================================

with open("assets/background.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode()

with open("assets/sidebar.png", "rb") as f:
    encoded_sidebar = base64.b64encode(f.read()).decode()


# =====================================
# CUSTOM WEBSITE DESIGN
# =====================================

st.markdown(f"""
<style>

/* =====================================
   MAIN WEBSITE BACKGROUND
===================================== */

.stApp {{
    background-image: url("data:image/png;base64,{encoded_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}


/* =====================================
   REMOVE TOP HEADER
===================================== */

header[data-testid="stHeader"] {{
    display: none;
}}

[data-testid="stToolbar"] {{
    display: none;
}}

[data-testid="stDecoration"] {{
    display: none;
}}


/* =====================================
   SIDEBAR BACKGROUND
===================================== */

[data-testid="stSidebar"] {{
    background-color: #501E17 !important;
    background-image: url("data:image/png;base64,{encoded_sidebar}");
    background-size: 100% auto;
    background-position: center bottom;
    background-repeat: no-repeat;
}}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
    padding-top: 0rem !important;
    background: transparent !important;
}}

[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
    display: none;
}}

[data-testid="stSidebar"] .block-container {{
    padding-top: 0.5rem !important;
}}


/* =====================================
   SIDEBAR TEXT
===================================== */

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #F5E9CE !important;
}}

[data-testid="stSidebar"] .stRadio label {{
    color: #F5E9CE !important;
}}

[data-testid="stSidebar"] .stRadio label:hover {{
    color: #D4382B !important;
}}


/* =====================================
   MAIN CONTENT
===================================== */

.block-container {{
    padding-top: 2rem;
}}

h1, h2 {{
    color: #4E1E17 !important;
}}

h3 {{
    color: #6E6F33 !important;
}}


/* =====================================
   METRICS
===================================== */

[data-testid="stMetricValue"] {{
    color: #B3343C !important;
}}

[data-testid="stMetricLabel"] {{
    color: #6E6F33 !important;
}}


/* =====================================
   TABLE BACKGROUND
===================================== */

[data-testid="stDataFrame"],
[data-testid="stTable"] {{
    background-color: #FFF6DE !important;
}}


/* =====================================
   CHART BACKGROUND
===================================== */

.stPlotlyChart {{
    background-color: #FFF6DE !important;
}}


/* =====================================
   BUTTONS
===================================== */

div[data-testid="stButton"] button,
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stDownloadButton"] button {{
    background-color: #6E6F33 !important;
    color: #FFF6DE !important;
    border: 1px solid #6E6F33 !important;
}}


/* =====================================
   BUTTON HOVER
===================================== */

div[data-testid="stButton"] button:hover,
div[data-testid="stFormSubmitButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover {{
    background-color: #4E1E17 !important;
    color: #FFF6DE !important;
    border-color: #4E1E17 !important;
}}


/* =====================================
   TABLE HEADERS
===================================== */

[data-testid="stTable"] th {{
    background-color: #6E6F33 !important;
    color: #FFF6DE !important;
}}

/* =====================================
   INPUT FIELDS
===================================== */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextArea"] textarea {{
    background-color: #FFF6DE !important;
    color: #4E1E17 !important;
    border: 1px solid #6E6F33 !important;
}}


/* =====================================
   DROPDOWN BOXES
===================================== */

[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {{
    background-color: #FFF6DE !important;
    color: #4E1E17 !important;
    border-color: #6E6F33 !important;
}}


/* =====================================
   DROPDOWN OPTIONS
===================================== */

[role="option"] {{
    background-color: #FFF6DE !important;
    color: #4E1E17 !important;
}}


/* =====================================
   FORM BOXES
===================================== */

[data-testid="stForm"] {{
    background-color: rgba(255, 246, 222, 0.85) !important;
    border: 1px solid #D8CFA0 !important;
    border-radius: 12px;
    padding: 20px;
}}


/* =====================================
   METRIC BOXES
===================================== */

[data-testid="stMetric"] {{
    background-color: rgba(255, 246, 222, 0.85) !important;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #D8CFA0;
}}


/* =====================================
   RADIO BUTTONS AND CHECKBOXES
===================================== */

[data-testid="stRadio"],
[data-testid="stCheckbox"] {{
    color: #4E1E17 !important;
}}


/* =====================================
   EXPANDER BOXES
===================================== */

[data-testid="stExpander"] {{
    background-color: rgba(255, 246, 222, 0.85) !important;
    border: 1px solid #D8CFA0 !important;
    border-radius: 10px;
}}


/* =====================================
   CHAT INPUT
===================================== */

[data-testid="stChatInput"] textarea {{
    background-color: #FFF6DE !important;
    color: #4E1E17 !important;
}}


/* =====================================
   PLACEHOLDER TEXT
===================================== */

input::placeholder,
textarea::placeholder {{
    color: #8A7560 !important;
    opacity: 1 !important;
}}


</style>
""", unsafe_allow_html=True)


# =====================================
# TABLE STYLING FUNCTION
# =====================================

def style_table(df):

    return (
        df.style
        .set_properties(**{
            "background-color": "#FFF6DE",
            "color": "#4E1E17"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#6E6F33"),
                    ("color", "#FFF6DE"),
                    ("font-weight", "bold")
                ]
            }
        ])
    )


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("মুদি দোকান")

st.sidebar.caption("Inventory Intelligence System")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Products",
        "Sales Entry",
        "Demand Prediction",
        "AI Sales Assistant",
        "Reports"
    ]
)


# -----------------------------------
# DASHBOARD
# -----------------------------------

if page == "Dashboard":

    st.title("📊 Dashboard")
    st.write("Monitor your shop's sales and inventory performance.")
    st.markdown("---")

    # Get actual data from the database
    products = get_products()
    sales = get_sales()
    low_stock = get_low_stock_products()

    # Product and inventory information
    total_products = len(products)
    total_stock = sum(p["stock_quantity"] for p in products)

    # Prepare sales information
    if sales:
        sales_df = pd.DataFrame(sales)
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"])

        today = pd.Timestamp.today().normalize()
        month_start = today.replace(day=1)

        today_sales = sales_df[
            sales_df["sold_at"].dt.normalize() == today
        ]

        monthly_sales = sales_df[
            sales_df["sold_at"] >= month_start
        ]

        today_revenue = today_sales["total_amount"].sum()
        monthly_revenue = monthly_sales["total_amount"].sum()
        total_units_sold = sales_df["quantity"].sum()

    else:
        sales_df = pd.DataFrame()
        today_revenue = 0
        monthly_revenue = 0
        total_units_sold = 0

    # Main metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Products", total_products)
    col2.metric("Available Stock", total_stock)
    col3.metric("Low Stock Products", len(low_stock))

    col4, col5, col6 = st.columns(3)

    col4.metric("Today's Revenue", f"৳{today_revenue:,.2f}")
    col5.metric("Monthly Revenue", f"৳{monthly_revenue:,.2f}")
    col6.metric("Total Units Sold", int(total_units_sold))

    st.markdown("---")

    # Sales trend
    st.subheader("📈 Sales Trend")

    if not sales_df.empty:

        daily_sales = (
            sales_df.groupby(sales_df["sold_at"].dt.date)["total_amount"]
            .sum()
        )

        last_7_days = pd.date_range(
            end=pd.Timestamp.today().normalize(),
            periods=7
        )

        chart_data = pd.DataFrame({
            "Date": last_7_days.date
        })

        chart_data["Revenue"] = [
            daily_sales.get(day, 0)
            for day in chart_data["Date"]
        ]

        fig = px.line(
            chart_data,
            x="Date",
            y="Revenue",
            markers=True,
            title="Daily Revenue — Last 7 Days"
        )

        fig.update_layout(
            paper_bgcolor="#FFF6DE",
            plot_bgcolor="#FFF6DE",
            font=dict(color="#4E1E17"),
            title_font=dict(color="#4E1E17"),
            xaxis=dict(gridcolor="#D8CFA0"),
            yaxis=dict(gridcolor="#D8CFA0")
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Your sales trend will appear here after you record sales.")

    st.markdown("---")

    # Best-selling products
    st.subheader("🏆 Best-Selling Products")

    if not sales_df.empty:

        best_sellers = (
            sales_df.groupby("product_name")["quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )

        best_sellers.columns = ["Product", "Units Sold"]

        st.dataframe(
            style_table(best_sellers),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No sales data available yet.")

    st.markdown("---")

    # Low stock alerts
    st.subheader("⚠️ Inventory Alerts")

    if low_stock:

        low_stock_df = pd.DataFrame(low_stock)

        display_columns = low_stock_df[
            ["name", "stock_quantity", "reorder_level"]
        ].copy()

        display_columns.columns = [
            "Product",
            "Current Stock",
            "Reorder Level"
        ]

        st.dataframe(
            style_table(display_columns),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.success("All products have sufficient stock.")


# -----------------------------------
# PRODUCT MANAGEMENT
# -----------------------------------

elif page == "Products":

    st.title("📦 Product Management")

    st.write(
        "Add, view, update and manage your shop products."
    )

    st.markdown("---")

    # -----------------------------------
    # ADD NEW PRODUCT
    # -----------------------------------

    st.subheader("➕ Add New Product")

    with st.form("add_product_form"):

        col1, col2 = st.columns(2)

        with col1:

            name = st.text_input(
                "Product Name",
                placeholder="Example: Rice"
            )

            category = st.selectbox(
                "Product Category",
                [
                    "Grocery",
                    "Dairy",
                    "Beverages",
                    "Snacks",
                    "Personal Care",
                    "Household",
                    "Other"
                ]
            )

            cost_price = st.number_input(
                "Buying Price (৳)",
                min_value=0.0,
                step=1.0
            )

        with col2:

            selling_price = st.number_input(
                "Selling Price (৳)",
                min_value=0.0,
                step=1.0
            )

            stock_quantity = st.number_input(
                "Available Stock",
                min_value=0,
                step=1
            )

            reorder_level = st.number_input(
                "Minimum Stock Alert Level",
                min_value=0,
                value=10,
                step=1
            )

        submitted = st.form_submit_button(
            "Add Product",
            use_container_width=True
        )

        if submitted:

            if not name.strip():

                st.error("Please enter a product name.")

            elif cost_price > selling_price:

                st.warning(
                    "Your buying price is higher than "
                    "your selling price. Please check."
                )

            else:

                try:

                    add_product(
                        name,
                        category,
                        cost_price,
                        selling_price,
                        stock_quantity,
                        reorder_level
                    )

                    st.success(
                        f"{name} added successfully!"
                    )

                    st.rerun()

                except Exception as e:

                    st.error(f"Error: {e}")

    st.markdown("---")

    # -----------------------------------
    # VIEW PRODUCTS
    # -----------------------------------

    st.subheader("📋 All Products")

    products = get_products()

    if products:

        products_df = pd.DataFrame(products)

        display_df = products_df[
            [
                "id",
                "name",
                "category",
                "cost_price",
                "selling_price",
                "stock_quantity",
                "reorder_level"
            ]
        ].copy()

        display_df.columns = [
            "ID",
            "Product",
            "Category",
            "Buying Price",
            "Selling Price",
            "Available Stock",
            "Alert Level"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Total products registered: {len(products)}"
        )

    else:

        st.info(
            "No products have been added yet. "
            "Use the form above to add your first product."
        )

    st.markdown("---")

    # -----------------------------------
    # EDIT PRODUCT
    # -----------------------------------

    st.subheader("✏️ Update Product")

    if products:

        product_options = {
            f"{p['name']} (ID: {p['id']})": p
            for p in products
        }

        selected_name = st.selectbox(
            "Select a product to edit",
            list(product_options.keys()),
            key="edit_product_select"
        )

        selected_product = product_options[selected_name]

        with st.form("edit_product_form"):

            col1, col2 = st.columns(2)

            with col1:

                edit_name = st.text_input(
                    "Product Name",
                    value=selected_product["name"]
                )

                edit_category = st.text_input(
                    "Category",
                    value=selected_product["category"]
                )

                edit_cost = st.number_input(
                    "Buying Price (৳)",
                    min_value=0.0,
                    value=float(selected_product["cost_price"]),
                    step=1.0
                )

            with col2:

                edit_selling = st.number_input(
                    "Selling Price (৳)",
                    min_value=0.0,
                    value=float(selected_product["selling_price"]),
                    step=1.0
                )

                edit_stock = st.number_input(
                    "Available Stock",
                    min_value=0,
                    value=int(selected_product["stock_quantity"]),
                    step=1
                )

                edit_reorder = st.number_input(
                    "Minimum Stock Alert",
                    min_value=0,
                    value=int(selected_product["reorder_level"]),
                    step=1
                )

            update_button = st.form_submit_button(
                "Save Changes",
                use_container_width=True
            )

            if update_button:

                if edit_cost > edit_selling:

                    st.warning(
                        "Buying price is higher than selling price."
                    )

                else:

                    try:

                        update_product(
                            selected_product["id"],
                            edit_name,
                            edit_category,
                            edit_cost,
                            edit_selling,
                            edit_stock,
                            edit_reorder
                        )

                        st.success(
                            "Product updated successfully!"
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(f"Error: {e}")

        # -----------------------------------
        # DELETE PRODUCT
        # -----------------------------------

        st.markdown("---")

        st.subheader("🗑️ Delete Product")

        delete_name = st.selectbox(
            "Select product to delete",
            list(product_options.keys()),
            key="delete_product_select"
        )

        if st.button(
            "Delete Selected Product",
            type="secondary"
        ):

            st.session_state["confirm_delete"] = True

        if st.session_state.get("confirm_delete", False):

            st.warning(
                f"Are you sure you want to delete {delete_name}?"
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button("Yes, Delete", type="primary"):

                    try:

                        delete_product(
                            product_options[delete_name]["id"]
                        )

                        st.session_state["confirm_delete"] = False

                        st.success("Product deleted successfully!")

                        st.rerun()

                    except ValueError as e:

                        st.session_state["confirm_delete"] = False

                        st.error(str(e))

            with col2:

                if st.button("Cancel"):

                    st.session_state["confirm_delete"] = False

                    st.rerun()

    else:

        st.info("Add a product before editing or deleting.")


# -----------------------------------
# SALES ENTRY
# -----------------------------------

elif page == "Sales Entry":

    st.title("🛒 Sales Management")

    st.write(
        "Record customer purchases and automatically "
        "update your inventory."
    )

    st.markdown("---")

    # -----------------------------------
    # GET AVAILABLE PRODUCTS
    # -----------------------------------

    products = get_products()

    if not products:

        st.warning(
            "No products available. "
            "Please add products from Product Management first."
        )

    else:

        # -----------------------------------
        # SALES ENTRY FORM
        # -----------------------------------

        st.subheader("➕ Record a New Sale")

        product_options = {
            f"{p['name']} (Stock: {p['stock_quantity']}) "
            f"(ID: {p['id']})": p
            for p in products
        }

        with st.form("sales_entry_form"):

            selected_product_name = st.selectbox(
                "Select Product",
                list(product_options.keys())
            )

            selected_product = product_options[
                selected_product_name
            ]

            st.write(
                f"**Selling Price:** "
                f"৳{selected_product['selling_price']:,.2f}"
            )

            st.write(
                f"**Available Stock:** "
                f"{selected_product['stock_quantity']} units"
            )

            quantity = st.number_input(
                "Quantity Sold",
                min_value=1,
                value=1,
                step=1
            )

            # Calculate expected amount

            total = (
                selected_product["selling_price"]
                * quantity
            )

            st.info(
                f"Expected Total: ৳{total:,.2f}"
            )

            submit_sale = st.form_submit_button(
                "Record Sale",
                use_container_width=True
            )

            if submit_sale:

                try:

                    result = record_sale(
                        selected_product["id"],
                        quantity
                    )

                    st.success(
                        "Sale recorded successfully!"
                    )

                    st.write(
                        f"**Total Amount:** "
                        f"৳{result['total_amount']:,.2f}"
                    )

                    st.write(
                        f"**Remaining Stock:** "
                        f"{result['remaining_stock']} units"
                    )

                except ValueError as e:

                    st.error(str(e))

                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )

        st.markdown("---")

        # -----------------------------------
        # SALES SUMMARY
        # -----------------------------------

        st.subheader("📊 Sales Summary")

        sales = get_sales()

        if sales:

            sales_df = pd.DataFrame(sales)

            sales_df["sold_at"] = pd.to_datetime(
                sales_df["sold_at"]
            )

            total_revenue = sales_df[
                "total_amount"
            ].sum()

            total_transactions = len(sales_df)

            total_units_sold = sales_df[
                "quantity"
            ].sum()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Revenue",
                f"৳{total_revenue:,.2f}"
            )

            col2.metric(
                "Transactions",
                total_transactions
            )

            col3.metric(
                "Units Sold",
                int(total_units_sold)
            )

            st.markdown("---")

            # -----------------------------------
            # SALES HISTORY
            # -----------------------------------

            st.subheader("📋 Sales History")

            display_sales = sales_df[
                [
                    "id",
                    "product_name",
                    "quantity",
                    "unit_price",
                    "total_amount",
                    "sold_at"
                ]
            ].copy()

            display_sales.columns = [
                "Sale ID",
                "Product",
                "Quantity",
                "Unit Price",
                "Total Amount",
                "Date and Time"
            ]

            display_sales["Date and Time"] = (
                display_sales["Date and Time"]
                .dt.strftime("%d-%m-%Y %I:%M %p")
            )

            st.dataframe(
                display_sales,
                use_container_width=True,
                hide_index=True
            )

            # -----------------------------------
            # FILTER SALES
            # -----------------------------------

            st.markdown("---")

            st.subheader("🔎 Filter Sales History")

            start_date = st.date_input(
                "Start Date",
                value=sales_df["sold_at"].min().date(),
                key="sales_start_date"
            )

            end_date = st.date_input(
                "End Date",
                value=sales_df["sold_at"].max().date(),
                key="sales_end_date"
            )

            if start_date > end_date:

                st.error(
                    "Start date cannot be after end date."
                )

            else:

                filtered_sales = sales_df[
                    (sales_df["sold_at"].dt.date >= start_date)
                    &
                    (sales_df["sold_at"].dt.date <= end_date)
                ]

                st.write(
                    f"Transactions found: "
                    f"{len(filtered_sales)}"
                )

                st.metric(
                    "Revenue in Selected Period",
                    f"৳{filtered_sales['total_amount'].sum():,.2f}"
                )

        else:

            st.info(
                "No sales have been recorded yet. "
                "Record your first sale using the form above."
            )


# -----------------------------------
# DEMAND PREDICTION
# -----------------------------------

elif page == "Demand Prediction":

    st.title("🤖 AI Demand Prediction")

    st.write(
        "Train a machine-learning model to estimate "
        "future product demand."
    )

    st.markdown("---")

    sales = get_sales()

    if not sales:

        st.warning(
            "No sales data available. Record some sales first."
        )

    else:

        dataset = prepare_sales_data(sales)

        st.subheader("📊 Historical Sales Data")

        st.dataframe(
            dataset,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.subheader("🧠 Train the AI Model")

        st.write(
            "The model learns from previous daily sales "
            "to estimate future demand."
        )

        st.info(
            "For this version, a product needs at least "
            "21 days of sales history before it can be trained."
        )

        if st.button(
            "🚀 Train Demand Prediction Model",
            use_container_width=True
        ):

            with st.spinner("Training machine-learning models..."):

                model_metrics, skipped_products = (
                    train_demand_models(dataset)
                )

            if model_metrics:

                st.success("Model training completed!")

                st.subheader("📈 Model Evaluation")

                metrics_df = pd.DataFrame([
                    {
                        "Product": details["product_name"],
                        "Mean Absolute Error": details["mae"],
                        "Training Records": details["training_records"]
                    }
                    for details in model_metrics.values()
                ])

                st.dataframe(
                    metrics_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning(
                    "No models were trained. "
                    "Your products need more sales history."
                )

            if skipped_products:

                st.warning(
                    "Not enough historical data for: "
                    + ", ".join(skipped_products)
                )

        st.markdown("---")

        st.subheader("🔮 Predict Tomorrow's Demand")

        products = get_products()

        product_options = {
            f"{p['name']} (ID: {p['id']})": p
            for p in products
            if p["id"] in dataset["product_id"].unique()
        }

        if product_options:

            selected_name = st.selectbox(
                "Select a product",
                list(product_options.keys())
            )

            selected_product = product_options[selected_name]

            if st.button(
                "Predict Tomorrow",
                use_container_width=True
            ):

                prediction = predict_tomorrow(
                    dataset,
                    selected_product["id"]
                )

                if prediction is None:

                    st.warning(
                        "No trained model is available for this product. "
                        "Train the model first and ensure the product "
                        "has enough sales history."
                    )

                else:

                    st.success(
                        f"Expected demand tomorrow: {prediction} units"
                    )

                    current_stock = selected_product["stock_quantity"]

                    st.metric(
                        "Current Available Stock",
                        f"{current_stock} units"
                    )

                    if current_stock < prediction:

                        st.error(
                            f"Possible shortage! You may need "
                            f"{prediction - current_stock} additional units."
                        )

                    else:

                        st.success(
                            "Current stock appears sufficient "
                            "for the predicted demand."
                        )

        else:

            st.info(
                "No products with recorded sales are available."
            )

        # -----------------------------------
        # SMART REORDER RECOMMENDATIONS
        # -----------------------------------

        st.markdown("---")

        st.subheader("📦 Smart Reorder Recommendations")

        st.write(
            "Get stock recommendations based on predicted "
            "demand for the next 7 days."
        )

        if product_options:

            recommendation_rows = []

            for product in product_options.values():

                recommendation = get_stock_recommendation(
                    dataset,
                    product["id"],
                    product["stock_quantity"]
                )

                if recommendation is not None:

                    recommendation_rows.append({
                        "Product": product["name"],
                        "Current Stock": product["stock_quantity"],
                        "Predicted 7-Day Demand": recommendation["weekly_demand"],
                        "Safety Stock": recommendation["safety_stock"],
                        "Reorder Point": recommendation["reorder_point"],
                        "Suggested Order": recommendation["suggested_order"]
                    })

            if recommendation_rows:

                recommendation_df = pd.DataFrame(
                    recommendation_rows
                )

                st.dataframe(
                    recommendation_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("---")

                st.subheader("🛒 Products That Need Reordering")

                reorder_products = recommendation_df[
                    recommendation_df["Suggested Order"] > 0
                ]

                if not reorder_products.empty:

                    st.dataframe(
                        reorder_products[
                            ["Product", "Current Stock", "Suggested Order"]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.success(
                        "No products currently need additional stock "
                        "based on the model's estimates."
                    )

            else:

                st.warning(
                    "No recommendations available. Train the models "
                    "and ensure products have sufficient sales history."
                )

        else:

            st.info(
                "Add products and record sales before generating "
                "recommendations."
            )


# -----------------------------------
# AI SALES ASSISTANT
# -----------------------------------

elif page == "AI Sales Assistant":

    st.title("💬 SmartStock AI Assistant")

    st.write(
        "Ask questions about your shop's sales, "
        "products, and inventory."
    )

    st.markdown("---")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested questions
    st.subheader("Try asking:")

    suggestions = [
        "Which products are low in stock?",
        "What is my total revenue?",
        "Which product sells the most?",
        "How many units are available in total?",
        "What is my monthly revenue?"
    ]

    for suggestion in suggestions:

        if st.button(suggestion, use_container_width=True):
            answer = answer_question(suggestion)

            st.session_state.chat_history.append({
                "question": suggestion,
                "answer": answer
            })

    st.markdown("---")

    # Display previous conversations
    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # User input
    user_question = st.chat_input(
        "Ask something about your shop..."
    )

    if user_question:

        answer = answer_question(user_question)

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer
        })

        st.rerun()

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# ==========================================
# REPORTS PAGE
# ==========================================

elif page == "Reports":

    st.title("Business Reports")
    st.write("View and download your shop's sales reports.")

    sales = get_sales()

    if not sales:

        st.info("No sales records available yet.")

    else:

        df = pd.DataFrame(sales)

        # Convert sales date
        df["sold_at"] = pd.to_datetime(df["sold_at"])

        st.subheader("Sales Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Transactions",
                len(df)
            )

        with col2:
            st.metric(
                "Total Units Sold",
                int(df["quantity"].sum())
            )

        with col3:
            st.metric(
                "Total Revenue",
                f"৳{df['total_amount'].sum():,.2f}"
            )

        st.divider()

        st.subheader("Sales Records")

        st.dataframe(
            df,
            use_container_width=True
        )

        # CSV Export
        csv_data = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Download Sales Report (CSV)",
            data=csv_data,
            file_name="smartstock_sales_report.csv",
            mime="text/csv"
        )

        st.success("Your sales report is ready to download.")
