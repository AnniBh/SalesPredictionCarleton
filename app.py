import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# Load the data
data = pd.read_csv("Updated_Sales_Data.csv")

# Convert the "Date" column to datetime format
data["Date"] = pd.to_datetime(data["Date"])
data["Year"] = data["Date"].dt.year
data["Year-Month"] = data["Date"].dt.to_period("M")

# Streamlit App
st.title("Sales Dashboard")

# Sidebar for filtering
st.sidebar.header("Filters")

# Year filter
current_year = datetime.now().year
year_filter = st.sidebar.selectbox(
    "Select Year:",
    options=sorted(data["Year"].unique(), reverse=True),
    index=sorted(data["Year"].unique(), reverse=True).index(current_year)
)

# Filter data by selected year
filtered_data_by_year = data[data["Year"] == year_filter]

# Category filter
category_filter = st.sidebar.multiselect(
    "Select Categories:",
    options=filtered_data_by_year["Category"].unique(),
    default=filtered_data_by_year["Category"].unique()
)

# Item filter
item_filter = st.sidebar.multiselect(
    "Select Items:",
    options=filtered_data_by_year["Item Name"].unique(),
    default=filtered_data_by_year["Item Name"].unique()
)

# Apply all filters
filtered_data = filtered_data_by_year[
    (filtered_data_by_year["Category"].isin(category_filter)) &
    (filtered_data_by_year["Item Name"].isin(item_filter))
]

# Metrics
st.header(f"Key Metrics for {year_filter}")
total_sales = len(filtered_data)
unique_items = filtered_data["Item Name"].nunique()
total_revenue = filtered_data["Price"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", total_sales)
col2.metric("Unique Items Sold", unique_items)
col3.metric("Total Revenue ($)", f"${total_revenue:,.2f}")

# Sales by Category
st.header("Sales by Category")
sales_by_category = filtered_data["Category"].value_counts()

if sales_by_category.empty:
    st.warning("No data available for the selected filters.")
else:
    fig, ax = plt.subplots()
    sales_by_category.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Sales by Category")
    ax.set_ylabel("Number of Sales")
    ax.set_xlabel("Category")
    st.pyplot(fig)

# Sales by Item
st.header("Sales by Item")
sales_by_item = filtered_data["Item Name"].value_counts()

if sales_by_item.empty:
    st.warning("No data available for the selected filters.")
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    sales_by_item.plot(kind="barh", ax=ax, color="lightgreen")
    ax.set_title("Sales by Item")
    ax.set_xlabel("Number of Sales")
    ax.set_ylabel("Item Name")
    st.pyplot(fig)

# Monthly Sales Line Chart
st.header("Monthly Sales")
monthly_sales = filtered_data.groupby("Year-Month").size()

if monthly_sales.empty:
    st.warning("No data available for the selected filters.")
else:
    fig, ax = plt.subplots()
    monthly_sales.plot(kind="line", ax=ax, marker="o", color="purple")
    ax.set_title("Monthly Sales Trend")
    ax.set_ylabel("Number of Sales")
    ax.set_xlabel("Month")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

# Average Calories per Category
st.header("Average Calories per Category")
calories_by_category = filtered_data.groupby("Category")["Calories"].mean()

if calories_by_category.empty:
    st.warning("No data available for the selected filters.")
else:
    fig, ax = plt.subplots()
    calories_by_category.plot(kind="bar", ax=ax, color="orange")
    ax.set_title("Average Calories per Category")
    ax.set_ylabel("Average Calories")
    ax.set_xlabel("Category")
    st.pyplot(fig)

# Raw Data
st.header(f"Raw Data for {year_filter}")
st.dataframe(filtered_data)
