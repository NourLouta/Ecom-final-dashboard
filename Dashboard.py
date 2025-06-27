
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

st.set_page_config(page_title="Pharmacy EDA Dashboard", layout="wide", page_icon="💊")

# Custom styling for Streamlit
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #D4F1F4;
    }
    .css-1d391kg, .css-1d391kg * {
        color: #004E64 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_excel("cleaned_final_project.xlsx")
    return df

if "filtered" not in st.session_state:
    df = load_data()
    st.session_state.df = df
    st.session_state.filtered = df

# Always assign from session (safe fallback)
df = st.session_state.df
df_filtered = st.session_state.filtered

# Load best model and label encoder
model = joblib.load("best_model.pkl")
le = joblib.load("label_encoder.pkl")

# Sidebar Filters
st.sidebar.title("🔎 Filters")
with st.sidebar.expander("📍 Refine your view"):
    selected_city = st.multiselect("🏩 Select City", options=sorted(df['Customer City'].unique()), default=df['Customer City'].unique())
    selected_category = st.multiselect("🧴 Select Category", options=sorted(df['Category'].unique()), default=df['Category'].unique())
    selected_gender = st.multiselect("👤 Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())

    if st.button("✅ Apply Filters"):
        st.session_state.filtered = df[
            (df['Customer City'].isin(selected_city)) &
            (df['Category'].isin(selected_category)) &
            (df['Gender'].isin(selected_gender))
        ].copy()

        # ✅ Immediately reassign df_filtered
        df_filtered = st.session_state.filtered

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["📊 EDA Visuals", "📅 Export & Summary", "📈 ML Predictions"])

with tab1:
    st.title("📊 EDA Dashboard — E-commerce Pharmacy")
    st.markdown("Explore 15+ interactive visual questions")

    st.write("### 🔍 Filtered Data Preview")
    st.dataframe(st.session_state.filtered.head(10), use_container_width=True)

    df_filtered = st.session_state.filtered

    # Visual 1: Top Cities
    st.subheader("🏬 Top 10 Cities by Total Sales")
    top_cities = df_filtered.groupby('Customer City')['Total sales'].sum().nlargest(10).reset_index()
    st.plotly_chart(px.bar(top_cities, x='Customer City', y='Total sales', color='Customer City'), use_container_width=True)

    # Visual 2: Category Sales
    st.subheader("📦 Sales by Product Category")
    cat_sales = df_filtered.groupby('Category')['Total sales'].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(cat_sales, x='Category', y='Total sales', color='Category'), use_container_width=True)

    # Visual 3: Gender
    st.subheader("👥 Sales by Gender")
    pie_data = df_filtered.groupby('Gender')['Total sales'].sum().reset_index()
    st.plotly_chart(px.pie(pie_data, names='Gender', values='Total sales', hole=0.4), use_container_width=True)

    # Visual 4: Monthly Trend
    st.subheader("📆 Monthly Sales Trend")
    monthly = df_filtered.groupby(['Year', 'Month_Name'])['Total sales'].sum().reset_index()
    st.plotly_chart(px.line(monthly, x='Month_Name', y='Total sales', color='Year', markers=True), use_container_width=True)

    # Visual 5: Category x Gender
    st.subheader("👥 Sales by Gender Across Categories")
    combo = df_filtered.groupby(['Gender', 'Category'])['Total sales'].sum().reset_index()
    st.plotly_chart(px.bar(combo, x='Category', y='Total sales', color='Gender', barmode='group'), use_container_width=True)

    # Visual 6: Sunburst
    st.subheader("🌍 Customer Distribution: City × Gender × Customer Type")
    dist = df_filtered.groupby(['Customer City', 'Gender', 'Customer Type']).size().reset_index(name='Count')
    st.plotly_chart(px.sunburst(dist, path=['Customer City', 'Gender', 'Customer Type'], values='Count'), use_container_width=True)

    # Visual 7: Invoice × Gender
    st.subheader("🧾 Invoice Type vs Gender")
    inv = df_filtered.groupby(['Invoice Type', 'Gender'])['Total sales'].mean().reset_index()
    st.plotly_chart(px.bar(inv, x='Invoice Type', y='Total sales', color='Gender', barmode='group'), use_container_width=True)

    # Visual 8: Bubble (QTY vs Sales)
    st.subheader("🔘 Quantity vs Sales by Category & Gender")
    bubble = df_filtered.groupby(['Category', 'Gender']).agg({'Total sales': 'sum', 'Items Qty Per Invoice': 'sum'}).reset_index()
    st.plotly_chart(px.scatter(bubble, x='Items Qty Per Invoice', y='Total sales', size='Total sales', color='Gender', hover_name='Category'), use_container_width=True)

    # Visual 9: Day of Week
    st.subheader("📅 Sales by Day of Week")
    dow = df_filtered.groupby('Day_Name')['Total sales'].sum().reset_index()
    st.plotly_chart(px.bar(dow, x='Day_Name', y='Total sales', color='Day_Name'), use_container_width=True)

    # Visual 10: Avg Qty per Category
    st.subheader("📊 Avg Qty per Invoice by Category")
    avg = df_filtered.groupby('Category')['Items Qty Per Invoice'].mean().reset_index()
    st.plotly_chart(px.bar(avg, x='Category', y='Items Qty Per Invoice', color='Category'), use_container_width=True)

    # Visual 11: Invoice × Customer Type
    st.subheader("💼 Avg Sales by Invoice & Customer Type")
    comb1 = df_filtered.groupby(['Invoice Type', 'Customer Type'])['Total sales'].mean().reset_index()
    st.plotly_chart(px.bar(comb1, x='Invoice Type', y='Total sales', color='Customer Type', barmode='group'), use_container_width=True)

    # Visual 12: Day × Customer Type
    st.subheader("📈 Avg Sales by Day & Customer Type")
    comb2 = df_filtered.groupby(['Day_Name', 'Customer Type'])['Total sales'].mean().reset_index()
    st.plotly_chart(px.bar(comb2, x='Day_Name', y='Total sales', color='Customer Type', barmode='group'), use_container_width=True)

    # Visual 13: Customer Type by Category & Gender
    st.subheader("📌 Customer Type by Category & Gender")
    sun2 = df_filtered.groupby(['Category', 'Gender', 'Customer Type']).size().reset_index(name='Count')
    st.plotly_chart(px.sunburst(sun2, path=['Category', 'Gender', 'Customer Type'], values='Count'), use_container_width=True)

    # Visual 14: Correlation Heatmap
    st.subheader("🧪 Correlation of Numeric Fields")
    numeric = df_filtered.select_dtypes(include='number')
    fig = px.imshow(numeric.corr(), text_auto=True, title='Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

    # Visual 15: Store-wise Performance
    st.subheader("🏪 Total Sales by Store")
    stores = df_filtered.groupby('Store ID')['Total sales'].sum().reset_index()
    st.plotly_chart(px.bar(stores, x='Store ID', y='Total sales', color='Store ID'), use_container_width=True)


    # Add Month, Year, Month_Name columns for tab1 if not exist
    if 'Month' not in df_filtered.columns:
        df_filtered['Trx Date'] = pd.to_datetime(df_filtered['Trx Date'], errors='coerce')
        df_filtered['Month'] = df_filtered['Trx Date'].dt.month
        df_filtered['Month_Name'] = df_filtered['Trx Date'].dt.strftime('%B')
        df_filtered['Year'] = df_filtered['Trx Date'].dt.year

with tab2:
    st.header("📅 Export Filtered Dataset")
    st.download_button("⬇ Download CSV", df_filtered.to_csv(index=False), file_name='filtered_sales_data.csv')
    st.success("You can now download the filtered data for reporting.")

with tab3:
    st.header("📈 Predict Customer Type")

    st.markdown("Use the form below to simulate an order and predict the likely **Customer Type**.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            category = st.selectbox("Category", sorted(df['Category'].unique()))
            brand_options = df[df['Category'] == category]['Brand'].unique()
            brand = st.selectbox("Brand", sorted(brand_options))
            city = st.selectbox("Customer City", sorted(df['Customer City'].unique()))
            invoice_type = st.selectbox("Invoice Type", df['Invoice Type'].unique())
            gender = st.selectbox("Gender", df['Gender'].unique())

        with col2:
            items_count = st.number_input("Invoices Items Count", min_value=1, max_value=50, value=3, step=1)
            qty_per_invoice = st.number_input("Items Qty Per Invoice", min_value=1, max_value=200, value=5, step=1)
            total_sales = st.number_input("Total Sales", min_value=0.0, value=100.0, step=1.0)
            order_value = total_sales / qty_per_invoice
            is_weekend = st.radio("Is Weekend?", [0, 1])
            high_price_flag = int(total_sales > df['Total sales'].mean())
            invoice_intensity = qty_per_invoice / items_count
            today = datetime.today()
            month = today.month
            month_name = today.strftime('%B')
            year = today.year

        submit = st.form_submit_button("🔮 Predict")

    if submit:
        input_df = pd.DataFrame([{
            'Customer City': city,
            'Category': category,
            'Brand': brand,
            'Invoice Type': invoice_type,
            'Gender': gender,
            'Invoices Items Count': items_count,
            'Items Qty Per Invoice': qty_per_invoice,
            'Total sales': total_sales,
            'OrderValuePerItem': order_value,
            'Is_Weekend': is_weekend,
            'High_Price_Flag': high_price_flag,
            'Invoice_Item_Intensity': invoice_intensity,
            'Month': month,
            'Month_Name': month_name,
            'Year': year
        }])

        # Predict using pipeline directly (with built-in preprocessing)
        prediction = model.predict(input_df)[0]
        pred_label = le.inverse_transform([prediction])[0]

        st.success(f"🌟 Predicted Customer Type: **{pred_label}**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Final Project — Noureldeen Loutah")
