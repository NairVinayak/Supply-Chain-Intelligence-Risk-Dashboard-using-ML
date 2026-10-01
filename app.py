%%writefile app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Supply Chain Intelligence Dashboard", layout="wide")

st.title("📦 Supply Chain Intelligence & Risk Dashboard using ML")
st.markdown("Analyzing delivery performance, demand forecasts, and risk across regions.")

# Load cleaned data (relative paths for GitHub/Streamlit Cloud)
df = pd.read_csv('Data/cleaned_data.csv')
region_clusters = pd.read_csv('Data/region_clusters.csv')

# Load model + columns
rf_model = joblib.load('models/rf_delay_model.pkl')
model_columns = joblib.load('models/model_columns.pkl')

# ---- KPI Section ----
st.header("📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
on_time_rate = (df['Late_delivery_risk'] == 0).mean() * 100
total_sales = df['Sales'].sum()
total_orders = df['Order Id'].nunique()
avg_delay = df['Days for shipping (real)'].sub(df['Days for shipment (scheduled)']).mean()

col1.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")
col2.metric("Total Sales", f"${total_sales/1e6:.1f}M")
col3.metric("Total Orders", f"{total_orders:,}")
col4.metric("Avg Shipping Delay", f"{avg_delay:.2f} days")

# ---- Charts Row ----
st.header("📈 Delivery Performance Analysis")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("On-Time Rate by Shipping Mode")
    shipping_otd = df.groupby('Shipping Mode')['Late_delivery_risk'].apply(lambda x: (x == 0).mean() * 100).sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(5,3.5))
    shipping_otd.plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_ylabel('On-Time Rate (%)', fontsize=9)
    ax1.axhline(on_time_rate, color='red', linestyle='--', linewidth=1, label='Avg')
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig1)

with chart_col2:
    st.subheader("Region Risk Map")
    cluster_labels = {0: "Low-Risk, Low-Volume", 1: "Standard Performance", 2: "High-Priority Risk", 3: "High-Risk, Low-Volume"}
    region_clusters['Risk Tier'] = region_clusters['Cluster'].map(cluster_labels)
    colors = {'Low-Risk, Low-Volume':'green', 'Standard Performance':'gray', 'High-Priority Risk':'red', 'High-Risk, Low-Volume':'orange'}
    fig2, ax2 = plt.subplots(figsize=(5,3.5))
    for tier in region_clusters['Risk Tier'].unique():
        subset = region_clusters[region_clusters['Risk Tier'] == tier]
        ax2.scatter(subset['on_time_rate'], subset['total_sales'], label=tier, s=60, color=colors.get(tier))
    ax2.set_xlabel('On-Time Rate (%)', fontsize=9)
    ax2.set_ylabel('Total Sales ($)', fontsize=9)
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig2)

# ---- Regional Risk Table ----
st.header("🌍 Regional Risk Breakdown")
st.dataframe(
    region_clusters[['Order Region', 'Risk Tier', 'on_time_rate', 'total_sales', 'order_count']]
    .sort_values('total_sales', ascending=False)
    .style.format({'on_time_rate': '{:.1f}%', 'total_sales': '${:,.0f}'}),
    height=300
)

# ---- Live Prediction Tool ----
st.header("🔮 Delay Risk Predictor")
st.markdown("Enter order details to predict the likelihood of a late delivery.")

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    shipping_mode = st.selectbox("Shipping Mode", df['Shipping Mode'].unique())
    order_region = st.selectbox("Order Region", df['Order Region'].unique())

with pred_col2:
    category = st.selectbox("Category", df['Category Name'].unique())
    scheduled_days = st.slider("Days for Shipment (Scheduled)", 0, 5, 2)

with pred_col3:
    quantity = st.number_input("Order Item Quantity", min_value=1, max_value=10, value=1)
    sales = st.number_input("Sales ($)", min_value=0.0, value=200.0)
    discount = st.slider("Discount Rate", 0.0, 0.5, 0.1)

if st.button("Predict Delay Risk"):
    input_dict = {
        'Order Item Quantity': quantity,
        'Sales': sales,
        'Days for shipment (scheduled)': scheduled_days,
        'Order Item Discount Rate': discount
    }
    input_df = pd.DataFrame([input_dict])

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    mode_col = f'Shipping Mode_{shipping_mode}'
    region_col = f'Order Region_{order_region}'
    category_col = f'Category Name_{category}'

    for col in [mode_col, region_col, category_col]:
        if col in input_df.columns:
            input_df[col] = 1

    input_df = input_df[model_columns]

    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK of late delivery — {probability*100:.1f}% probability")
    else:
        st.success(f"✅ LOW RISK of late delivery — {probability*100:.1f}% probability of being late")
