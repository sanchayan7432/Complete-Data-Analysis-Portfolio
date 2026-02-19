import streamlit as st
import os

st.set_page_config(
    page_title="Data Analysis Portfolio",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analysis & Machine Learning Portfolio")

st.markdown("""
Welcome to my portfolio showcasing real-world analytics projects.
""")


st.markdown("""
Developed by: SANCHAYAN GHOSH.
""")

st.sidebar.title("📁 PORTFOLIO")

project = st.sidebar.radio(
    "Select Project",
    [
        "🏠 Home",
        "📊 Sales Analysis",
        "🏥 Healthcare Analysis",
        "⚽ Sports Analytics",
        "💰 Financial Analysis",
        "🛒 E-commerce Churn Analysis"
    ]
)

#base_path = os.path.dirname(os.path.dirname(__file__))
base_path = os.path.dirname(__file__)


# =========================
# HOME
# =========================
if project == "🏠 Home":
    st.header("👋 Overview")

    st.write("""
This portfolio demonstrates:

✔ Data Cleaning  
✔ Exploratory Data Analysis  
✔ Visualization  
✔ Machine Learning  
✔ Business Insights  
""")

# =========================
# SALES
# =========================
elif project == "📊 Sales Analysis":
    st.header("📊 Sales Analysis")

    st.write("**Objective:** Analyze sales trends and performance.")

    img_path = os.path.join(base_path, "project1_sales_analysis", "visualizations")

    if os.path.exists(img_path):
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                st.image(os.path.join(img_path, img))

# =========================
# HEALTHCARE
# =========================
elif project == "🏥 Healthcare Analysis":
    st.header("🏥 Healthcare Analysis")

    img_path = os.path.join(base_path, "project2_healthcare_analysis", "visualizations")

    if os.path.exists(img_path):
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                st.image(os.path.join(img_path, img))

# =========================
# SPORTS
# =========================
elif project == "⚽ Sports Analytics":
    st.header("⚽ Sports Analytics")

    img_path = os.path.join(base_path, "project3_sports_analytics", "visualizations")

    if os.path.exists(img_path):
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                st.image(os.path.join(img_path, img))

# =========================
# FINANCIAL
# =========================
elif project == "💰 Financial Analysis":
    st.header("💰 Financial Analysis")

    img_path = os.path.join(base_path, "project4_financial_analysis", "visualizations")

    if os.path.exists(img_path):
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                st.image(os.path.join(img_path, img))

# =========================
# CHURN
# =========================
elif project == "🛒 E-commerce Churn Analysis":
    st.header("🛒 E-commerce Churn Analysis")

    img_path = os.path.join(base_path, "project5_ecommerce_analytics", "visualizations")

    if os.path.exists(img_path):
        for img in os.listdir(img_path):
            if img.endswith(".png"):
                st.image(os.path.join(img_path, img))


