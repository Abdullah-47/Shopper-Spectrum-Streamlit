# =============================================================================
# STREAMLIT APP FOR CUSTOMER SEGMENTATION AND PRODUCT RECOMMENDATION
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation & Product Recommendation",
    page_icon="🛍️",
    layout="wide"
)

# Load saved models
@st.cache_resource
def load_models():
    """Load all saved models"""
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('recommendation_model.pkl', 'rb') as f:
        model_knn, product_customer_matrix = pickle.load(f)
    with open('cluster_names.pkl', 'rb') as f:
        cluster_names = pickle.load(f)
    
    return kmeans, scaler, model_knn, product_customer_matrix, cluster_names

# Load data
@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('online_retail.csv')
        return df
    except:
        st.error("Dataset not found. Please ensure 'online_retail.csv' is in the same directory.")
        return None

# Main app
def main():
    st.title("🛍️ Customer Segmentation & Product Recommendation System")
    st.markdown("---")
    
    # Load models and data
    kmeans, scaler, model_knn, product_customer_matrix, cluster_names = load_models()
    df = load_data()
    
    if df is None:
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", 
                           ["📊 Dashboard", "👥 Customer Segmentation", "🛒 Product Recommendations"])
    
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "👥 Customer Segmentation":
        show_customer_segmentation(kmeans, scaler, cluster_names)
    elif page == "🛒 Product Recommendations":
        show_product_recommendations(model_knn, product_customer_matrix, df)

def show_dashboard(df):
    """Show main dashboard with analytics"""
    st.header("📊 Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
    with col2:
        st.metric("Total Products", f"{df['StockCode'].nunique():,}")
    with col3:
        st.metric("Total Transactions", f"{df['InvoiceNo'].nunique():,}")
    with col4:
        total_revenue = (df['Quantity'] * df['UnitPrice']).sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    st.markdown("---")
    
    # Transaction volume by country
    st.subheader("Transaction Volume by Country")
    country_volume = df.groupby('Country').size().sort_values(ascending=False).head(10)
    
    fig_country = px.bar(x=country_volume.values, y=country_volume.index, 
                        orientation='h', title='Top 10 Countries by Transaction Volume')
    fig_country.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_country, use_container_width=True)
    
    # Top selling products
    st.subheader("Top Selling Products")
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    
    fig_products = px.bar(x=top_products.values, y=top_products.index, 
                         orientation='h', title='Top 10 Products by Quantity Sold')
    fig_products.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_products, use_container_width=True)
    
    # Purchase trends over time
    st.subheader("Purchase Trends Over Time")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    monthly_trends = df.groupby('Month')['Quantity'].sum()
    
    fig_trends = px.line(x=monthly_trends.index.astype(str), y=monthly_trends.values, 
                        title='Monthly Purchase Trends')
    fig_trends.update_xaxes(title="Month")
    fig_trends.update_yaxes(title="Total Quantity Sold")
    st.plotly_chart(fig_trends, use_container_width=True)

def show_customer_segmentation(kmeans, scaler, cluster_names):
    """Show customer segmentation module"""
    st.header("👥 Customer Segmentation")
    
    st.markdown("""
    ### Predict Customer Segment
    Enter customer's RFM values to predict their segment:
    """)
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.number_input("Recency (days since last purchase)", 
                                 min_value=0, max_value=1000, value=30)
    with col2:
        frequency = st.number_input("Frequency (number of purchases)", 
                                   min_value=1, max_value=1000, value=5)
    with col3:
        monetary = st.number_input("Monetary (total spend)", 
                                 min_value=0.0, max_value=100000.0, value=500.0)
    
    if st.button("Predict Cluster", type="primary"):
        # Preprocess input
        recency_sqrt = np.sqrt(recency)
        frequency_log = np.log1p(frequency)
        monetary_log = np.log1p(monetary)
        
        # Scale the features
        input_scaled = scaler.transform([[recency_sqrt, frequency_log, monetary_log]])
        
        # Predict cluster
        cluster = kmeans.predict(input_scaled)[0]
        cluster_name = cluster_names.get(cluster, "Unknown Segment")
        
        # Display result
        st.success(f"**Predicted Segment: {cluster_name}**")
        
        # Show segment characteristics
        st.subheader("Segment Characteristics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Recency", f"{recency} days")
        with col2:
            st.metric("Frequency", f"{frequency} purchases")
        with col3:
            st.metric("Monetary", f"${monetary:.2f}")
    
    st.markdown("---")
    
    # Cluster visualization
    st.subheader("Customer Segment Overview")
    
    # Create sample data for visualization
    sample_data = {
        'Recency': np.random.randint(1, 365, 100),
        'Frequency': np.random.randint(1, 50, 100),
        'Monetary': np.random.randint(10, 5000, 100)
    }
    sample_df = pd.DataFrame(sample_data)
    
    # Preprocess and predict
    sample_df['Recency_sqrt'] = np.sqrt(sample_df['Recency'])
    sample_df['Frequency_log'] = np.log1p(sample_df['Frequency'])
    sample_df['Monetary_log'] = np.log1p(sample_df['Monetary'])
    
    sample_scaled = scaler.transform(sample_df[['Recency_sqrt', 'Frequency_log', 'Monetary_log']])
    sample_df['Cluster'] = kmeans.predict(sample_scaled)
    sample_df['ClusterName'] = sample_df['Cluster'].map(cluster_names)
    
    # 3D scatter plot
    fig = px.scatter_3d(sample_df, x='Recency', y='Frequency', z='Monetary', 
                       color='ClusterName', title='Customer Segments Visualization')
    st.plotly_chart(fig, use_container_width=True)

def show_product_recommendations(model_knn, product_customer_matrix, df):
    """Show product recommendation module"""
    st.header("🛒 Product Recommendations")
    
    st.markdown("""
    ### Get Product Recommendations
    Enter a product name or code to get similar product recommendations:
    """)
    
    # Product selection
    product_options = df['Description'].unique().tolist()
    selected_product = st.selectbox("Select a product:", product_options)
    
    # Get product code
    product_code = df[df['Description'] == selected_product]['StockCode'].iloc[0]
    
    if st.button("Get Recommendations", type="primary"):
        # Get recommendations
        from main import get_product_recommendations
        recommendations = get_product_recommendations(product_code, model_knn, product_customer_matrix)
        
        if recommendations:
            st.success(f"**Top 5 recommendations for {selected_product}:**")
            
            # Display recommendations in cards
            cols = st.columns(5)
            for i, (rec_product, similarity) in enumerate(recommendations[:5]):
                with cols[i]:
                    # Get product description
                    product_desc = df[df['StockCode'] == rec_product]['Description'].iloc[0]
                    
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; text-align: center;">
                        <h4>{product_desc}</h4>
                        <p>Code: {rec_product}</p>
                        <p>Similarity: {similarity:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("No recommendations found for this product.")
    
    st.markdown("---")
    
    # Product similarity heatmap
    st.subheader("Product Similarity Matrix (Sample)")
    
    # Create a sample similarity matrix for visualization
    sample_products = product_customer_matrix.index[:20]  # First 20 products
    sample_matrix = product_customer_matrix.loc[sample_products]
    
    # Calculate cosine similarity for sample
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(sample_matrix)
    
    # Create heatmap
    fig = px.imshow(similarity_matrix, 
                    x=sample_products, 
                    y=sample_products,
                    title="Product Similarity Heatmap (Sample)")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()