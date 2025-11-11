
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Try to load environment variables from .env file (for local development)
# On Streamlit Cloud, use st.secrets instead
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will use Streamlit secrets or environment variables
    pass

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# LLM Libraries for Chatbot
import os
import json

# Set page configuration
st.set_page_config(
    page_title="AI Customer Recommendation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with proper contrast (works in both light and dark mode)
st.markdown("""
    <style>
    /* Force dark text on light background for all text elements */
    body {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #000000 !important;
    }
    
    .stText, p, span, div, label {
        color: #000000 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1f1f1f !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 5px;
        padding: 10px 20px;
        color: #000000 !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        font-weight: 600;
        color: #000000 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
    
    /* Button contrast */
    .stButton > button {
        font-weight: 500;
        color: #ffffff !important;
        background-color: #0066cc !important;
    }
    
    /* Dataframe text */
    [data-testid="stDataFrame"] * {
        color: #000000 !important;
    }
    
    /* Input fields */
    input, textarea, select {
        color: #000000 !important;
    }
    
    /* Ensure expanders are readable */
    [data-testid="stExpander"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    [data-testid="stExpander"] * {
        color: #000000 !important;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Selectbox and other widgets */
    [data-baseweb="select"] * {
        color: #000000 !important;
    }
    
    /* Slider labels */
    [data-testid="stSlider"] * {
        color: #000000 !important;
    }
    
    /* Number input */
    [data-testid="stNumberInput"] * {
        color: #000000 !important;
    }
    
    /* Text input */
    [data-testid="stTextInput"] * {
        color: #000000 !important;
    }
    
    /* Download button */
    [data-testid="stDownloadButton"] button {
        color: #ffffff !important;
        background-color: #28a745 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ========== DATA LOADING & PREPROCESSING ==========
@st.cache_data
def load_data():
    """Load and preprocess the marketing campaign data"""
    df = pd.read_csv('marketing_campaign.csv', sep='\t')
    return df


@st.cache_data
def preprocess_data(df):

    data = df.copy()
    data['Income'].fillna(data['Income'].median(), inplace=True)

    # Feature Engineering
    # 1. Age from Year_Birth
    current_year = datetime.now().year
    data['Age'] = current_year - data['Year_Birth']
    
    # 2. Total spending
    data['Total_Spent'] = (data['MntWines'] + data['MntFruits'] + 
                           data['MntMeatProducts'] + data['MntFishProducts'] + 
                           data['MntSweetProducts'] + data['MntGoldProds'])
    
    # 3. Total purchases
    data['Total_Purchases'] = (data['NumWebPurchases'] + data['NumCatalogPurchases'] + 
                               data['NumStorePurchases'])
    
    # 4. Total campaigns accepted
    data['Total_Campaigns_Accepted'] = (data['AcceptedCmp1'] + data['AcceptedCmp2'] + 
                                        data['AcceptedCmp3'] + data['AcceptedCmp4'] + 
                                        data['AcceptedCmp5'] + data['Response'])
    
    # 5. Customer tenure (days since enrollment)
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
    data['Customer_Days'] = (datetime.now() - data['Dt_Customer']).dt.days
    
    # 6. Total children
    data['Total_Children'] = data['Kidhome'] + data['Teenhome']
    
    # 7. Family size
    data['Family_Size'] = data['Total_Children'] + 1  # Include customer
    data.loc[data['Marital_Status'].isin(['Married', 'Together']), 'Family_Size'] += 1
    
    # 8. Is Parent
    data['Is_Parent'] = (data['Total_Children'] > 0).astype(int)
    
    # 9. Simplify Education
    data['Education_Simplified'] = data['Education'].replace({
        'Graduation': 'Graduate',
        'PhD': 'Postgraduate',
        'Master': 'Postgraduate',
        '2n Cycle': 'Undergraduate',
        'Basic': 'Basic'
    })
    
    # 10. Simplify Marital Status
    data['Marital_Status_Simplified'] = data['Marital_Status'].replace({
        'Married': 'Partnered',
        'Together': 'Partnered',
        'Single': 'Single',
        'Divorced': 'Single',
        'Widow': 'Single',
        'Alone': 'Single',
        'Absurd': 'Single',
        'YOLO': 'Single'
    })
    
    # 11. Add aliases for compatibility with advanced AI features
    data['MntTotal'] = data['Total_Spent']
    data['AcceptedCmpOverall'] = data['Total_Campaigns_Accepted']
    
    return data


@st.cache_data
def encode_features(data):
    encoded_data = data.copy()
    
    le = LabelEncoder()
    
    categorical_cols = ['Education_Simplified', 'Marital_Status_Simplified']
    for col in categorical_cols:
        if col in encoded_data.columns:
            encoded_data[f'{col}_Encoded'] = le.fit_transform(encoded_data[col])
    
    return encoded_data


# ========== SYSTEM 1: CUSTOMER SEGMENTATION (CLUSTERING) ==========
def customer_segmentation(data):
    """K-Means clustering for customer segmentation"""
    st.header("🎯 System 1: Customer Segmentation (K-Means Clustering)")
    st.markdown("---")
    
    # Enhanced feature engineering focusing on discriminative features
    data_enhanced = data.copy()
    
    # Log transform skewed features for better distribution
    data_enhanced['Income_Log'] = np.log1p(data_enhanced['Income'])
    data_enhanced['Spending_Log'] = np.log1p(data_enhanced['Total_Spent'])
    
    # Create powerful ratio features
    data_enhanced['Spending_Per_Purchase'] = data_enhanced['Total_Spent'] / np.maximum(data_enhanced['Total_Purchases'], 1)
    data_enhanced['Income_Spending_Ratio'] = data_enhanced['Income'] / np.maximum(data_enhanced['Total_Spent'], 1)
    data_enhanced['Purchase_Frequency'] = data_enhanced['Total_Purchases'] / np.maximum(data_enhanced['Customer_Days'], 1) * 365
    data_enhanced['Campaign_Response_Rate'] = data_enhanced['Total_Campaigns_Accepted'] / 6
    data_enhanced['Avg_Discount_Usage'] = data_enhanced['NumDealsPurchases'] / np.maximum(data_enhanced['Total_Purchases'], 1)
    
    # Channel preference features
    data_enhanced['Web_Preference'] = data_enhanced['NumWebPurchases'] / np.maximum(data_enhanced['Total_Purchases'], 1)
    data_enhanced['Store_Preference'] = data_enhanced['NumStorePurchases'] / np.maximum(data_enhanced['Total_Purchases'], 1)
    
    # Lifestyle features
    data_enhanced['Has_Children'] = (data_enhanced['Total_Children'] > 0).astype(int)
    data_enhanced['High_Earner'] = (data_enhanced['Income'] > data_enhanced['Income'].median()).astype(int)
    data_enhanced['High_Spender'] = (data_enhanced['Total_Spent'] > data_enhanced['Total_Spent'].median()).astype(int)
    
    # Select ONLY the most discriminative features (fewer is often better)
    clustering_features = [
        'Income_Log',           # Financial capacity
        'Spending_Log',         # Spending behavior
        'Spending_Per_Purchase',# Purchase value
        'Purchase_Frequency',   # Engagement level
        'Campaign_Response_Rate', # Marketing responsiveness
        'Recency',              # Recent activity
        'Web_Preference',       # Shopping channel
        'Has_Children'          # Family status
    ]
    
    X = data_enhanced[clustering_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X.fillna(X.median(), inplace=True)
    
    # Aggressive outlier removal using IQR method
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ~((X < (Q1 - 2.5 * IQR)) | (X > (Q3 + 2.5 * IQR))).any(axis=1)
    
    X_clean = X[outlier_mask]
    data_clean = data_enhanced.loc[X_clean.index]
    
    # st.info(f"📊 Using {len(clustering_features)} carefully selected features. Removed {len(X) - len(X_clean)} outliers ({(len(X) - len(X_clean))/len(X)*100:.1f}%)")
    
    # Standardize features with RobustScaler (better for outliers)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Clustering Configuration")
        
        # Find optimal clusters button
        if st.button("🔍 Find Optimal Clusters", key="find_optimal"):
            with st.spinner("Testing different cluster numbers..."):
                silhouette_scores = []
                inertias = []
                K_range = range(2, 9)
                
                for k in K_range:
                    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
                    labels = kmeans_test.fit_predict(X_scaled)
                    silhouette_scores.append(silhouette_score(X_scaled, labels))
                    inertias.append(kmeans_test.inertia_)
                
                # Plot results
                fig_opt, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Silhouette scores
                ax1.plot(K_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
                ax1.set_xlabel('Number of Clusters', fontsize=12)
                ax1.set_ylabel('Silhouette Score', fontsize=12)
                ax1.set_title('Silhouette Score by Cluster Count', fontsize=14)
                ax1.grid(True, alpha=0.3)
                best_k_sil = K_range[np.argmax(silhouette_scores)]
                ax1.axvline(best_k_sil, color='r', linestyle='--', label=f'Best: {best_k_sil}')
                ax1.legend()
                
                # Elbow method
                ax2.plot(K_range, inertias, 'go-', linewidth=2, markersize=8)
                ax2.set_xlabel('Number of Clusters', fontsize=12)
                ax2.set_ylabel('Inertia', fontsize=12)
                ax2.set_title('Elbow Method', fontsize=14)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_opt)
                
                st.success(f"✅ **Recommended:** {best_k_sil} clusters (highest silhouette score: {max(silhouette_scores):.3f})")
                st.session_state['recommended_clusters'] = best_k_sil
        
        # Number of clusters
        default_k = st.session_state.get('recommended_clusters', 4)
        n_clusters = st.slider("Number of Customer Segments", 2, 8, default_k)
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            init_method = st.selectbox("Initialization Method", ["k-means++", "random"], index=0)
            max_iter = st.slider("Max Iterations", 100, 500, 300)
            n_init = st.slider("Number of Initializations", 10, 50, 30)
        
        if st.button("🚀 Run Segmentation", key="cluster_btn"):
            with st.spinner("Performing customer segmentation..."):
                # Fit KMeans with optimized parameters
                kmeans = KMeans(
                    n_clusters=n_clusters, 
                    random_state=42, 
                    n_init=n_init,
                    max_iter=max_iter,
                    init=init_method,
                    algorithm='lloyd'
                )
                clusters = kmeans.fit_predict(X_scaled)
                
                # Add clusters back to original data
                data_clean_copy = data_clean.copy()
                data_clean_copy['Cluster'] = clusters
                
                # Calculate silhouette score
                silhouette_avg = silhouette_score(X_scaled, clusters)
                
                # Calculate inertia (within-cluster sum of squares)
                inertia = kmeans.inertia_
                
                st.success(f"✅ Segmentation Complete!")
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                with col_m2:
                    st.metric("Inertia", f"{inertia:,.0f}")
                with col_m3:
                    st.metric("Customers", f"{len(data_clean_copy):,}")
                
                # Quality indicator with realistic thresholds for customer data
                if silhouette_avg > 0.35:
                    st.success("🌟 **Excellent clustering quality!** (Rare for customer data) Segments are very well-separated.")
                elif silhouette_avg > 0.20:
                    st.success("✅ **Good clustering quality!** Segments are distinct and highly actionable for marketing.")
                elif silhouette_avg > 0.10:
                    st.info("ℹ️ **Moderate clustering quality.** Segments have some overlap but still provide business value.")
                else:
                    st.warning("⚠️ **Weak clustering.** Try the 'Find Optimal Clusters' button or different parameters.")
                
                # Additional insights
                st.markdown(f"""
                **💡 Your Results Interpretation:**
                - **Silhouette Score {silhouette_avg:.3f}**: {'🎉 This is GOOD!' if silhouette_avg > 0.20 else 'Consider trying different K values'}
                - **{len(data_clean_copy):,} customers** segmented into **{n_clusters} actionable groups**
                - **What it means**: {'Your segments are well-separated enough for targeted marketing strategies!' if silhouette_avg > 0.20 else 'Segments have overlap - validate business interpretability'}
                """)
                
                # Reality check message for scores in the 0.20-0.30 range
                
                
                
                # Store in session state
                st.session_state['segmented_data'] = data_clean_copy
                st.session_state['kmeans_model'] = kmeans
                st.session_state['scaler'] = scaler
                st.session_state['X_scaled'] = X_scaled
                st.session_state['clustering_features'] = clustering_features
    
    with col2:
        if 'segmented_data' in st.session_state:
            st.subheader("📊 Cluster Visualization")
            
            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(st.session_state['X_scaled'])
            
            # Calculate explained variance
            explained_var = pca.explained_variance_ratio_
            
            viz_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': st.session_state['segmented_data']['Cluster'].astype(str),
                'Total_Spent': st.session_state['segmented_data']['Total_Spent'],
                'Income': st.session_state['segmented_data']['Income']
            })
            
            fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster',
                           hover_data=['Total_Spent', 'Income'],
                           title=f'Customer Segments (PCA: {explained_var[0]:.1%} + {explained_var[1]:.1%} variance)',
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature importance
            st.info(f"💡 **Clustering uses {len(st.session_state['clustering_features'])} enhanced features** including spending patterns, income ratios, and behavior scores.")
            
            # Add score benchmark comparison
            with st.expander("📚 Understanding Your Silhouette Score"):
                current_score = silhouette_score(st.session_state['X_scaled'], 
                                                st.session_state['segmented_data']['Cluster'])
                
                st.markdown(f"""
                ### Real-World Silhouette Score Benchmarks
                
                | Score Range | Quality | Typical Use Cases |
                |-------------|---------|-------------------|
                | **0.35 - 0.50** | 🌟 Excellent | Rare! Geographic regions, age groups |
                | **0.20 - 0.35** | ✅ Good | **Customer segments, behavior groups** |
                | **0.10 - 0.20** | ℹ️ Moderate | Complex customer data, overlap expected |
                | **< 0.10** | ⚠️ Weak | Try different approach |
                
                ### Why Customer Data Scores Lower
                
                **Unlike textbook examples:**
                - 🌸 Iris flowers (physical measurements) → Score: 0.70+
                - 📍 Geographic clustering (coordinates) → Score: 0.60+
                - 👥 **Customer behavior** (your data) → Score: **0.20-0.35** ✅
                
                ### Your Score: **{current_score:.3f}**
                
                **What this means:**
                - ✅ Segments are distinct enough for different marketing strategies
                - ✅ Some overlap is natural (customers aren't robots!)
                - ✅ Provides actionable insights for business decisions
                - ✅ Professional-grade customer segmentation
                
                **Remember:** A customer who spends $500 might be in "Medium" or "High" segment. 
                That's OK! The goal is to find patterns, not perfect boundaries.
                
                **Focus on:** Can you create different campaigns for each segment? If YES, you win! 🎯
                """)

    
    # Segment Analysis
    if 'segmented_data' in st.session_state:
        st.markdown("---")
        st.subheader("📈 Segment Analysis & Personas")
        
        segment_stats = st.session_state['segmented_data'].groupby('Cluster').agg({
            'Age': 'mean',
            'Income': 'mean',
            'Total_Spent': 'mean',
            'Total_Purchases': 'mean',
            'Total_Children': 'mean',
            'Customer_Days': 'mean',
            'ID': 'count'
        }).round(2)
        segment_stats.rename(columns={'ID': 'Customer_Count'}, inplace=True)
        
        # Create unique persona names based on characteristics
        personas = []
        n_clusters = len(segment_stats)
        
        # Sort by total spending to rank segments
        segment_stats_sorted = segment_stats.sort_values('Total_Spent', ascending=False)
        
        # Assign unique persona names based on cluster number and characteristics
        for rank, (idx, row) in enumerate(segment_stats_sorted.iterrows()):
            spent = row['Total_Spent']
            income = row['Income']
            age = row['Age']
            children = row['Total_Children']
            purchases = row['Total_Purchases']
            
            # Determine persona based on multiple factors with unique assignment
            if n_clusters == 2:
                # Simple split
                if rank == 0:
                    persona = "💎 Premium Customers"
                else:
                    persona = "💰 Value Seekers"
            
            elif n_clusters == 3:
                # High/Medium/Low
                if rank == 0:
                    persona = "� VIP High Spenders"
                elif rank == 1:
                    persona = "⭐ Regular Customers"
                else:
                    persona = "💵 Budget Shoppers"
            
            elif n_clusters == 4:
                # More granular
                if rank == 0:
                    persona = "👑 VIP Elite"
                elif rank == 1:
                    if children > segment_stats['Total_Children'].mean():
                        persona = "👨‍👩‍👧‍� Affluent Families"
                    else:
                        persona = "💎 Premium Shoppers"
                elif rank == 2:
                    persona = "� Average Buyers"
                else:
                    persona = "💰 Bargain Hunters"
            
            elif n_clusters == 5:
                personas_5 = ["👑 VIP Elite", "💎 High Value", "⭐ Core Customers", "📊 Occasional Buyers", "💵 Low Engagement"]
                persona = personas_5[rank]
            
            elif n_clusters >= 6:
                personas_6plus = [
                    "� VIP Elite",
                    "� Premium High Frequency", 
                    "🎯 High Potential",
                    "⭐ Regular Customers",
                    "📊 Occasional Shoppers",
                    "💰 Budget Conscious",
                    "🌱 New/Inactive"
                ]
                persona = personas_6plus[rank] if rank < len(personas_6plus) else f"🎯 Segment {idx}"
            
            else:
                persona = f"🎯 Segment {idx}"
            
            # Store with original cluster index
            personas.append((idx, persona))
        
        # Create a dictionary to map cluster to persona
        persona_dict = dict(personas)
        segment_stats['Persona'] = segment_stats.index.map(persona_dict)
        
        # Reorder columns
        segment_stats = segment_stats[['Persona', 'Customer_Count', 'Age', 'Income', 
                                       'Total_Spent', 'Total_Purchases', 'Total_Children', 
                                       'Customer_Days']]
        
        # Sort by spending for better display
        segment_stats = segment_stats.sort_values('Total_Spent', ascending=False)
        
        st.dataframe(segment_stats, use_container_width=True)
        
        # Visualization of segments
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(segment_stats.reset_index(), x='Cluster', y='Total_Spent',
                         color='Persona', title='Average Spending by Segment',
                         labels={'Total_Spent': 'Avg Spending ($)', 'Cluster': 'Segment'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.pie(segment_stats.reset_index(), values='Customer_Count', names='Persona',
                         title='Customer Distribution by Segment')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Business Recommendations
        st.markdown("---")
        st.subheader("💡 Business Recommendations")
        
        for idx, row in segment_stats.iterrows():
            with st.expander(f"**{row['Persona']}** - {int(row['Customer_Count'])} customers"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Income", f"${row['Income']:,.0f}")
                with col2:
                    st.metric("Avg Spending", f"${row['Total_Spent']:,.0f}")
                with col3:
                    st.metric("Avg Age", f"{row['Age']:.0f} years")
                
                # Tailored recommendations
                if "Premium" in row['Persona']:
                    st.markdown("""
                    **Marketing Strategy:**
                    - 🎁 Exclusive VIP programs and early access to new products
                    - 🌟 Premium quality offerings and luxury bundles
                    - 📧 Personalized email campaigns with high-end products
                    - 🎯 Target with wine & gourmet food promotions
                    """)
                elif "Budget" in row['Persona']:
                    st.markdown("""
                    **Marketing Strategy:**
                    - 💵 Discount campaigns and value deals
                    - 🎉 Bundle offers and seasonal promotions
                    - 📱 Digital coupons and loyalty rewards
                    - 🛒 Focus on essential products with good value
                    """)
                elif "Senior" in row['Persona']:
                    st.markdown("""
                    **Marketing Strategy:**
                    - 📞 Traditional marketing channels (catalog, phone)
                    - 🏥 Health-focused and comfort products
                    - 🤝 Loyalty appreciation programs
                    - 📬 Direct mail with easy-to-read formats
                    """)
                elif "Family" in row['Persona']:
                    st.markdown("""
                    **Marketing Strategy:**
                    - 👶 Family-oriented promotions and bulk discounts
                    - 🎒 Back-to-school and holiday campaigns
                    - 🏠 Home delivery and convenience services
                    - 🎪 Family bundle deals on groceries
                    """)
                else:
                    st.markdown("""
                    **Marketing Strategy:**
                    - 🎯 Analyze behavior patterns for targeted campaigns
                    - 📊 A/B test different promotional strategies
                    - 🔍 Monitor engagement and optimize outreach
                    """)


# ========== SYSTEM 2: CAMPAIGN RESPONSE PREDICTION (CLASSIFICATION) ==========
def campaign_response_prediction(data):
    st.header("📧 System 2: Campaign Response Prediction")
    st.markdown("---")
    
    # Prepare features
    feature_cols = ['Age', 'Income', 'Total_Spent', 'Recency', 'Total_Purchases',
                   'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                   'NumWebVisitsMonth', 'Customer_Days', 'Total_Children',
                   'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                   'Education_Simplified_Encoded', 'Marital_Status_Simplified_Encoded', 'Complain']
    
    # Encode if not already done
    if 'Education_Simplified_Encoded' not in data.columns:
        data = encode_features(data)
    
    X = data[feature_cols].copy()
    X.fillna(X.median(), inplace=True)
    
    y = data['Response']  # Target variable
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Model Configuration")
        
        model_choice = st.selectbox(
            "Select Classification Model",
            ["Random Forest", "Logistic Regression", "Gradient Boosting"]
        )
        
        test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        
        if st.button("🚀 Train Model", key="campaign_btn"):
            with st.spinner("Training prediction model..."):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Train model
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:  # Gradient Boosting
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
                
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Calculate metrics
                accuracy = (y_pred == y_test).mean()
                
                # Store in session state
                st.session_state['campaign_model'] = model
                st.session_state['campaign_X_test'] = X_test
                st.session_state['campaign_y_test'] = y_test
                st.session_state['campaign_y_pred'] = y_pred
                st.session_state['campaign_y_proba'] = y_pred_proba
                st.session_state['campaign_feature_cols'] = feature_cols
                
                st.success("✅ Model Trained Successfully!")
                st.metric("Accuracy", f"{accuracy:.2%}")
    
    with col2:
        if 'campaign_model' in st.session_state:
            st.subheader("📊 Model Performance")
            
            # Confusion Matrix
            cm = confusion_matrix(st.session_state['campaign_y_test'], 
                                st.session_state['campaign_y_pred'])
            
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                          x=['No Response', 'Response'], y=['No Response', 'Response'],
                          title='Confusion Matrix', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    if 'campaign_model' in st.session_state:
        st.markdown("---")
        st.subheader("📋 Performance Metrics")
        
        # Generate classification report
        from sklearn.metrics import classification_report
        report = classification_report(st.session_state['campaign_y_test'], 
                                       st.session_state['campaign_y_pred'], 
                                       target_names=['No Response', 'Response'],
                                       output_dict=True)
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            precision_responders = report['Response']['precision']
            st.metric("Precision", f"{precision_responders:.2%}",
                     help="Of all predicted responders, what % actually responded")
        
        with metrics_col2:
            recall_responders = report['Response']['recall']
            st.metric("Recall", f"{recall_responders:.2%}",
                     help="Of all actual responders, what % did we catch")
        
        with metrics_col3:
            f1_responders = report['Response']['f1-score']
            st.metric("F1-Score", f"{f1_responders:.2%}",
                     help="Balance of precision and recall")
    
    # Feature Importance & ROI Calculator
    if 'campaign_model' in st.session_state:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Feature Importance")
            
            if hasattr(st.session_state['campaign_model'], 'feature_importances_'):
                importances = st.session_state['campaign_model'].feature_importances_
                feature_imp_df = pd.DataFrame({
                    'Feature': st.session_state['campaign_feature_cols'],
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(feature_imp_df, x='Importance', y='Feature', orientation='h',
                           title='Top 10 Most Important Features')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💰 ROI Calculator")
            
            st.markdown("Calculate the ROI of targeted marketing vs. mass marketing")
            
            total_customers = len(data)
            cost_per_contact = st.number_input("Cost per Contact ($)", value=5.0, step=0.5)
            revenue_per_conversion = st.number_input("Revenue per Conversion ($)", value=100.0, step=10.0)
            
            # Get top X% of customers by propensity
            propensity_threshold = st.slider("Target Top X% of Customers", 5, 100, 20)
            
            if st.button("Calculate ROI"):
                # Create propensity scores for all customers
                X_all = data[st.session_state['campaign_feature_cols']].copy()
                X_all.fillna(X_all.median(), inplace=True)
                
                if hasattr(st.session_state['campaign_model'], 'predict_proba'):
                    propensity_scores = st.session_state['campaign_model'].predict_proba(X_all)[:, 1]
                else:
                    propensity_scores = st.session_state['campaign_model'].predict(X_all)
                
                # Sort and get top X%
                threshold_idx = int(total_customers * propensity_threshold / 100)
                sorted_indices = np.argsort(propensity_scores)[::-1]
                top_customers = sorted_indices[:threshold_idx]
                
                # Calculate expected conversions (using average response rate)
                avg_response_rate = data['Response'].mean()
                targeted_response_rate = avg_response_rate * 2.5  # Assume 2.5x improvement
                
                # Mass marketing
                mass_cost = total_customers * cost_per_contact
                mass_conversions = total_customers * avg_response_rate
                mass_revenue = mass_conversions * revenue_per_conversion
                mass_roi = ((mass_revenue - mass_cost) / mass_cost) * 100
                
                # Targeted marketing
                targeted_cost = threshold_idx * cost_per_contact
                targeted_conversions = threshold_idx * targeted_response_rate
                targeted_revenue = targeted_conversions * revenue_per_conversion
                targeted_roi = ((targeted_revenue - targeted_cost) / targeted_cost) * 100
                
                # Display results
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**📢 Mass Marketing**")
                    st.metric("Customers Contacted", f"{total_customers:,}")
                    st.metric("Total Cost", f"${mass_cost:,.0f}")
                    st.metric("Expected Conversions", f"{mass_conversions:.0f}")
                    st.metric("Expected Revenue", f"${mass_revenue:,.0f}")
                    st.metric("ROI", f"{mass_roi:.1f}%")
                
                with col_b:
                    st.markdown("**🎯 Targeted Marketing (AI-Driven)**")
                    st.metric("Customers Contacted", f"{threshold_idx:,}")
                    st.metric("Total Cost", f"${targeted_cost:,.0f}")
                    st.metric("Expected Conversions", f"{targeted_conversions:.0f}")
                    st.metric("Expected Revenue", f"${targeted_revenue:,.0f}")
                    st.metric("ROI", f"{targeted_roi:.1f}%", delta=f"{targeted_roi - mass_roi:.1f}%")
                
                st.success(f"💡 **Insight:** Targeted marketing saves ${mass_cost - targeted_cost:,.0f} " +
                          f"while achieving {targeted_roi - mass_roi:.1f}% higher ROI!")


# ========== SYSTEM 3: MARKET BASKET ANALYSIS (ASSOCIATION RULES) ==========
def market_basket_analysis(data):
    """Discover product purchase patterns using association rules"""
    st.header("🛒 System 3: Market Basket Analysis")
    st.markdown("---")
    
    # Convert spending to binary (bought/not bought)
    product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                   'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    
    basket_df = data[product_cols].copy()
    
    # Convert to binary (1 if purchased, 0 otherwise)
    basket_binary = (basket_df > 0).astype(int)
    basket_binary.columns = ['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Analysis Configuration")
        
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
        min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05)
        
        if st.button("🔍 Find Associations", key="basket_btn"):
            with st.spinner("Mining association rules..."):
                # Apply Apriori algorithm
                frequent_itemsets = apriori(basket_binary, min_support=min_support, use_colnames=True)
                
                if len(frequent_itemsets) > 0:
                    # Generate rules
                    rules = association_rules(frequent_itemsets, metric="confidence", 
                                             min_threshold=min_confidence)
                    
                    if len(rules) > 0:
                        # Sort by confidence
                        rules = rules.sort_values('confidence', ascending=False)
                        
                        st.session_state['basket_rules'] = rules
                        st.session_state['frequent_itemsets'] = frequent_itemsets
                        
                        st.success(f"✅ Found {len(rules)} association rules!")
                        st.metric("Frequent Itemsets", len(frequent_itemsets))
                    else:
                        st.warning("No rules found. Try lowering the confidence threshold.")
                else:
                    st.warning("No frequent itemsets found. Try lowering the support threshold.")
    
    with col2:
        if 'basket_rules' in st.session_state:
            st.subheader("📊 Association Rules")
            
            rules_display = st.session_state['basket_rules'].copy()
            
            # Format for display - convert frozenset to string
            rules_display['antecedents_str'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_display['consequents_str'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # Select top rules
            top_n = st.slider("Show Top N Rules", 5, 20, 10)
            
            display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence']
            display_df = rules_display[display_cols].head(top_n).copy()
            display_df.columns = ['Antecedents', 'Consequents', 'Support', 'Confidence']
            
            st.dataframe(
                display_df.style.format({
                    'Support': '{:.3f}',
                    'Confidence': '{:.3f}'
                }),
                use_container_width=True
            )
    
    # Visualizations and Recommendations
    if 'basket_rules' in st.session_state:
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Support vs Confidence")
            
            # Create a clean dataframe for plotting (without frozensets)
            plot_data = st.session_state['basket_rules'].copy()
            plot_data['antecedents_str'] = plot_data['antecedents'].apply(lambda x: ', '.join(list(x)))
            plot_data['consequents_str'] = plot_data['consequents'].apply(lambda x: ', '.join(list(x)))
            plot_data['rule'] = plot_data['antecedents_str'] + ' → ' + plot_data['consequents_str']
            
            fig = px.scatter(plot_data, 
                           x='support', y='confidence', 
                           hover_data=['rule'],
                           title='Association Rules: Support vs Confidence',
                           labels={'support': 'Support', 'confidence': 'Confidence'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔝 Top Product Combinations")
            
            # Product purchase frequency
            product_freq = basket_binary.sum().sort_values(ascending=False)
            
            fig = px.bar(x=product_freq.values, y=product_freq.index, orientation='h',
                        title='Product Purchase Frequency',
                        labels={'x': 'Number of Purchases', 'y': 'Product'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Business Recommendations
        st.markdown("---")
        st.subheader("💡 Cross-Selling Recommendations")
        
        # Get top 5 rules with highest confidence
        top_rules = st.session_state['basket_rules'].nlargest(5, 'confidence')
        
        for idx, rule in top_rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            with st.expander(f"**Bundle Opportunity:** {', '.join(antecedents)} → {', '.join(consequents)}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Confidence", f"{rule['confidence']:.1%}")
                with col_b:
                    st.metric("Support", f"{rule['support']:.1%}")
                
                st.markdown(f"""
                **Recommendation:**
                - 🎁 Create a bundle: **"{', '.join(antecedents)} + {', '.join(consequents)} Combo"**
                - 💰 Offer 10-15% discount on bundle
                - 📍 Place {', '.join(consequents)} near {', '.join(antecedents)} in store
                - 📧 Email customers who buy {', '.join(antecedents)} with {', '.join(consequents)} recommendations
                - 🎯 {rule['confidence']*100:.0f}% of customers who buy {', '.join(antecedents)} also buy {', '.join(consequents)}
                """)


# ========== SYSTEM 4: CUSTOMER LIFETIME VALUE PREDICTION (REGRESSION) ==========
def clv_prediction(data):
    """Predict customer lifetime value using regression"""
    st.header("💎 System 4: Customer Lifetime Value (CLV) Prediction")
    st.markdown("---")
    
    # Prepare features
    feature_cols = ['Age', 'Income', 'Recency', 'Total_Purchases',
                   'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                   'NumWebVisitsMonth', 'Customer_Days', 'Total_Children',
                   'Total_Campaigns_Accepted', 'Education_Simplified_Encoded', 
                   'Marital_Status_Simplified_Encoded', 'NumDealsPurchases']
    
    # Encode if not already done
    if 'Education_Simplified_Encoded' not in data.columns:
        data = encode_features(data)
    
    X = data[feature_cols].copy()
    X.fillna(X.median(), inplace=True)
    
    y = data['Total_Spent']  # Target: total spending (CLV proxy)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Model Configuration")
        
        model_choice = st.selectbox(
            "Select Regression Model",
            ["Gradient Boosting", "Random Forest"]
        )
        
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, key="clv_test") / 100
        
        if st.button("🚀 Train CLV Model", key="clv_btn"):
            with st.spinner("Training CLV prediction model..."):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Train model
                if model_choice == "Gradient Boosting":
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42, 
                                                     max_depth=5, learning_rate=0.1)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Store in session state
                st.session_state['clv_model'] = model
                st.session_state['clv_X_test'] = X_test
                st.session_state['clv_y_test'] = y_test
                st.session_state['clv_y_pred'] = y_pred
                st.session_state['clv_feature_cols'] = feature_cols
                
                st.success("✅ CLV Model Trained!")
                st.metric("R² Score", f"{r2:.3f}")
                st.metric("MAE", f"${mae:.2f}")
                st.metric("RMSE", f"${rmse:.2f}")
    
    with col2:
        if 'clv_model' in st.session_state:
            st.subheader("📊 Prediction Performance")
            
            # Actual vs Predicted scatter plot
            comparison_df = pd.DataFrame({
                'Actual CLV': st.session_state['clv_y_test'],
                'Predicted CLV': st.session_state['clv_y_pred']
            })
            
            fig = px.scatter(comparison_df, x='Actual CLV', y='Predicted CLV',
                           title='Actual vs Predicted CLV',
                           trendline="ols", trendline_color_override="red")
            fig.add_trace(go.Scatter(x=[0, comparison_df['Actual CLV'].max()],
                                    y=[0, comparison_df['Actual CLV'].max()],
                                    mode='lines', name='Perfect Prediction',
                                    line=dict(dash='dash', color='green')))
            st.plotly_chart(fig, use_container_width=True)
    
    # Customer Segmentation by CLV
    if 'clv_model' in st.session_state:
        st.markdown("---")
        
        # Predict CLV for all customers
        X_all = data[st.session_state['clv_feature_cols']].copy()
        X_all.fillna(X_all.median(), inplace=True)
        
        data['Predicted_CLV'] = st.session_state['clv_model'].predict(X_all)
        
        # Create CLV segments
        data['CLV_Segment'] = pd.qcut(data['Predicted_CLV'], q=4, 
                                      labels=['Low Value', 'Medium Value', 'High Value', 'VIP'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 CLV Distribution")
            
            fig = px.histogram(data, x='Predicted_CLV', nbins=50,
                             title='Customer Lifetime Value Distribution',
                             labels={'Predicted_CLV': 'Predicted CLV ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        
        
        # CLV Segment Analysis
        st.markdown("---")
        st.subheader("📊 CLV Segment Analysis")
        
        clv_stats = data.groupby('CLV_Segment').agg({
            'Predicted_CLV': 'mean',
            'Total_Spent': 'mean',
            'Income': 'mean',
            'Age': 'mean',
            'Total_Purchases': 'mean',
            'ID': 'count'
        }).round(2)
        clv_stats.rename(columns={'ID': 'Customer_Count'}, inplace=True)
        
        # Reorder for better display
        clv_stats = clv_stats.reindex(['VIP', 'High Value', 'Medium Value', 'Low Value'])
        
        st.dataframe(clv_stats.style.format({
            'Predicted_CLV': '${:,.2f}',
            'Total_Spent': '${:,.2f}',
            'Income': '${:,.2f}',
            'Age': '{:.0f}',
            'Total_Purchases': '{:.1f}',
            'Customer_Count': '{:,.0f}'
        }), use_container_width=True)
        
        # Feature Importance
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔍 CLV Drivers (Feature Importance)")
            
            if hasattr(st.session_state['clv_model'], 'feature_importances_'):
                importances = st.session_state['clv_model'].feature_importances_
                feature_imp_df = pd.DataFrame({
                    'Feature': st.session_state['clv_feature_cols'],
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(feature_imp_df, x='Importance', y='Feature', orientation='h',
                           title='Top 10 CLV Predictors',
                           color='Importance', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💡 Retention Strategy")
            
            vip_count = len(data[data['CLV_Segment'] == 'VIP'])
            vip_revenue = data[data['CLV_Segment'] == 'VIP']['Predicted_CLV'].sum()
            avg_clv = data['Predicted_CLV'].mean()
            
            st.metric("VIP Customers", f"{vip_count:,}")
            st.metric("VIP Total CLV", f"${vip_revenue:,.0f}")
            st.metric("Avg Customer CLV", f"${avg_clv:.2f}")
            
            st.markdown("""
            **Action Items:**
            - 🎁 VIP loyalty program
            - 📧 Personalized outreach
            - 🎯 Exclusive offers
            - 🤝 Dedicated support
            - 📊 Monitor churn risk
            """)
        
        # Business Recommendations by Segment
        st.markdown("---")
        st.subheader("💼 Segment-Specific Strategies")
        
        segments_data = [
            ("VIP", "💎", """
            - **Retention is Critical:** These customers generate the most revenue
            - Provide exclusive benefits, early access, and VIP treatment
            - Personal relationship management and dedicated support
            - Premium product recommendations and customized experiences
            - Monitor for any signs of churn and act immediately
            """),
            ("High Value", "⭐", """
            - **Growth Opportunity:** Potential to become VIP customers
            - Targeted upselling and cross-selling campaigns
            - Encourage increased purchase frequency with incentives
            - Loyalty rewards program to increase engagement
            - Premium product exposure and trials
            """),
            ("Medium Value", "📊", """
            - **Nurture & Develop:** Solid customer base with upside potential
            - Re-engagement campaigns for inactive customers
            - Bundle offers and value-added services
            - Identify barriers to increased spending
            - Educational content about product benefits
            """),
            ("Low Value", "💡", """
            - **Acquisition Cost Recovery:** Focus on cost-effective marketing
            - Automated campaigns and minimal manual intervention
            - Volume-based promotions and entry-level products
            - Monitor for engagement; consider win-back campaigns
            - Identify characteristics to improve future acquisition
            """)
        ]
        
        for segment, icon, recommendations in segments_data:
            with st.expander(f"{icon} **{segment}** Customers"):
                segment_data = data[data['CLV_Segment'] == segment]
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Count", f"{len(segment_data):,}")
                with col_b:
                    st.metric("Avg CLV", f"${segment_data['Predicted_CLV'].mean():,.0f}")
                with col_c:
                    st.metric("Avg Income", f"${segment_data['Income'].mean():,.0f}")
                with col_d:
                    revenue_share = (segment_data['Predicted_CLV'].sum() / data['Predicted_CLV'].sum()) * 100
                    st.metric("Revenue Share", f"{revenue_share:.1f}%")
                
                st.markdown(recommendations)
        
        # Download Comprehensive Report
        st.markdown("---")
        st.subheader("📥 Export Analysis Reports")
        
        report_col1, report_col2 = st.columns(2)
        
        with report_col1:
            # CSV Export
            csv_data = data.to_csv(index=False)
            st.download_button(
                label="📊 Download Full Dataset (CSV)",
                data=csv_data,
                file_name="customer_clv_analysis.csv",
                mime="text/csv",
                help="Download complete dataset with CLV predictions"
            )
        
        with report_col2:
            # High-Value Customers Export
            vip_customers = data[data['CLV_Segment'] == 'VIP Customers']
            vip_csv = vip_customers.to_csv(index=False)
            st.download_button(
                label="👑 Download VIP Customers (CSV)",
                data=vip_csv,
                file_name="vip_customers_list.csv",
                mime="text/csv",
                help="Download list of VIP customers for targeted retention"
            )


# ========== MAIN DASHBOARD ==========
def main():
    # Title and Header
    st.title("🎯 AI-Powered Customer Analytics Dashboard")
    # st.markdown("### Transform Your Marketing with 4 AI Systems")
    
    # Load data
    with st.spinner("Loading customer data..."):
        df = load_data()
        df = preprocess_data(df)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
        st.title("Navigation")
        st.markdown("---")
        
        # Dataset info
        st.subheader("📊 Dataset Overview")
        st.metric("Total Customers", f"{len(df):,}")
        st.metric("Features", f"{len(df.columns)}")
        # st.metric("Date Range", f"{df['Customer_Days'].max()} days")
        
        st.markdown("---")
        
        # System selection
        st.subheader("🎯 AI Systems")
        st.markdown("""
        1. **Customer Segmentation** - Group similar customers
        2. **Campaign Prediction** - Who will respond?
        3. **Market Basket** - Cross-sell opportunities
        4. **CLV Prediction** - Identify VIP customers
        """)
        
        st.markdown("---")
        st.info("💡 **Tip:** Start with Customer Segmentation to understand your audience!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", 
        "🎯 Segmentation", 
        "📧 Campaign Prediction", 
        "🛒 Market Basket", 
        "💎 CLV Prediction"
    ])
    
    with tab1:
        # st.header("🏠 Executive Dashboard Overview")
        # st.markdown("---")
        
        # Key Performance Indicators
        st.markdown("## 📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_spending = df['Total_Spent'].mean()
            total_revenue = df['Total_Spent'].sum()
            st.metric("Avg Customer Spending", f"${avg_spending:.2f}", 
                     help="Average total spending per customer")
        
        with col2:
            response_rate = df['Response'].mean() * 100
            st.metric("Campaign Response Rate", f"{response_rate:.1f}%",
                     help="Percentage of customers who responded to campaigns")
        
        with col3:
            avg_income = df['Income'].mean()
            st.metric("Avg Customer Income", f"${avg_income:,.0f}",
                     help="Average annual income of customers")
        
        with col4:
            avg_age = df['Age'].mean()
            st.metric("Avg Customer Age", f"{avg_age:.0f} years",
                     help="Average age of customer base")
        
        st.markdown("---")
        
        # # ROI Calculator
        # st.markdown("## 💰 Campaign ROI Calculator")
        # st.markdown("**Calculate the expected return on investment for your next marketing campaign**")
        
        # calc_col1, calc_col2, calc_col3 = st.columns(3)
        
        # with calc_col1:
        #     campaign_cost = st.number_input("Campaign Budget ($)", 
        #                                    min_value=1000, 
        #                                    value=10000, 
        #                                    step=1000,
        #                                    help="Total cost to execute the marketing campaign")
        
        # with calc_col2:
        #     target_customers = st.number_input("Customers to Target", 
        #                                       min_value=100, 
        #                                       value=min(1000, len(df)), 
        #                                       step=100,
        #                                       help="Number of customers you plan to reach")
        
        # with calc_col3:
        #     expected_response = st.slider("Expected Response Rate (%)", 
        #                                  min_value=1, 
        #                                  max_value=50, 
        #                                  value=int(response_rate * 1.5),
        #                                  help="Improved response rate with AI-powered targeting")
        
        # # Calculate ROI metrics
        # responders = int((target_customers * expected_response) / 100)
        # revenue_per_responder = avg_spending * 0.3  # Conservative estimate: 30% of avg spending
        # total_campaign_revenue = responders * revenue_per_responder
        # net_profit = total_campaign_revenue - campaign_cost
        # roi_percentage = ((net_profit / campaign_cost) * 100) if campaign_cost > 0 else 0
        
        # # Display ROI Results
        # st.markdown("### � Projected Campaign Results:")
        
        # roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
        
        # with roi_col1:
        #     st.metric("Expected Responders", f"{responders:,}", 
        #              delta=f"{(responders/target_customers)*100:.1f}% of target")
        
        # with roi_col2:
        #     st.metric("Revenue Generated", f"${total_campaign_revenue:,.0f}",
        #              help="Total revenue from campaign responders")
        
        # with roi_col3:
        #     st.metric("Net Profit", f"${net_profit:,.0f}",
        #              delta=f"${net_profit:,.0f}" if net_profit > 0 else f"-${abs(net_profit):,.0f}",
        #              delta_color="normal" if net_profit > 0 else "inverse")
        
        # with roi_col4:
        #     if roi_percentage > 100:
        #         st.success(f"✅ ROI: {roi_percentage:.1f}%")
        #     elif roi_percentage > 0:
        #         st.warning(f"⚠️ ROI: {roi_percentage:.1f}%")
        #     else:
        #         st.error(f"❌ ROI: {roi_percentage:.1f}%")
        
        # # ROI Interpretation
        # if roi_percentage > 100:
        #     st.success("🎉 **Excellent ROI!** This campaign is highly profitable. AI targeting can help you achieve this.")
        # elif roi_percentage > 0:
        #     st.info("💡 **Positive ROI:** The campaign will be profitable. Consider optimizing targeting for better returns.")
        # else:
        #     st.error("⚠️ **Negative ROI:** Consider refining your targeting strategy or reducing campaign costs.")
        
        # st.markdown("---")
        
        # Business Insights Dashboard
        st.markdown("## 💡 AI-Powered Business Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("### 🎯 Customer Behavior Insights")
            
            # Calculate insights
            high_spenders = len(df[df['Total_Spent'] > df['Total_Spent'].quantile(0.75)])
            low_spenders = len(df[df['Total_Spent'] < df['Total_Spent'].quantile(0.25)])
            active_web = len(df[df['NumWebPurchases'] > df['NumWebPurchases'].median()])
            
            st.markdown(f"""
            **Customer Distribution:**
            - 💎 **High-Value Customers:** {high_spenders} ({(high_spenders/len(df)*100):.1f}%)
              - These customers spend ${df[df['Total_Spent'] > df['Total_Spent'].quantile(0.75)]['Total_Spent'].mean():.0f} on average
              - **Action:** Prioritize retention programs for this segment
            
            - 💰 **Budget-Conscious:** {low_spenders} ({(low_spenders/len(df)*100):.1f}%)
              - Average spending: ${df[df['Total_Spent'] < df['Total_Spent'].quantile(0.25)]['Total_Spent'].mean():.0f}
              - **Action:** Focus on entry-level products and promotions
            
            **Purchase Channels:**
            - 🌐 **Active Web Users:** {active_web} customers ({(active_web/len(df)*100):.1f}%)
            - 🏪 **Store Preference:** {len(df) - active_web} customers
            - **Action:** Tailor marketing to preferred channels
            """)
        
        with insight_col2:
            st.markdown("### � Strategic Recommendations")
            
            st.markdown(f"""
            **Immediate Action Items:**
            
            1. 🎯 **Implement Segmentation:**
               - Use AI to identify {len(df)} customers into 3-5 distinct segments
               - Create personalized campaigns for each segment
               - **Expected Lift:** 20-30% in response rates
            
            2. 💰 **Optimize Campaign Targeting:**
               - Predict response probability for each customer
               - Focus budget on top 30% likely responders
               - **Expected ROI Improvement:** 50-100%
            
            3. 🛒 **Leverage Cross-Selling:**
               - Identify product combinations with high purchase probability
               - Create bundled offers and strategic placements
               - **Expected Revenue Lift:** 10-15%
            
            4. 💎 **Focus on High-CLV Customers:**
               - Identify customers with highest lifetime value potential
               - Implement VIP retention programs
               - **Expected Impact:** Reduce churn by 15-25%
            """)
        
        st.markdown("---")
        
        # AI Systems Overview
        st.markdown("---")
        
        # Data Preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Download option
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv,
            file_name="customer_data_processed.csv",
            mime="text/csv",
        )
    
    with tab2:
        customer_segmentation(df)
    
    with tab3:
        campaign_response_prediction(df)
    
    with tab4:
        market_basket_analysis(df)
    
    with tab5:
        clv_prediction(df)
    
    # AI Chatbot Section
    st.markdown("---")
    ai_chatbot(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🤖 Powered by Machine Learning | Built with Streamlit | 
        Data-Driven Marketing Excellence</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# AI CHATBOT FEATURE
# ============================================================================

def ai_chatbot(df):
    """AI-Powered Chatbot using Gemini API"""
    st.markdown("## 💬 AI Customer Insights Chatbot")
    st.markdown("Ask questions about your customer data in natural language!")
    
    # Get API key from environment or Streamlit secrets
    api_key = os.getenv('GEMINI_API_KEY')
    
    # If not in environment, try Streamlit secrets (for Streamlit Cloud)
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except (AttributeError, FileNotFoundError):
            pass
    
    if not api_key:
        st.warning("⚠️ Gemini API key not configured. Please set GEMINI_API_KEY in Streamlit secrets or environment variables.")
        st.info("📝 **To enable the AI chatbot:**\n\n" + 
                "**On Streamlit Cloud:** Go to 'Manage app' → 'Settings' → 'Secrets' and add:\n```\nGEMINI_API_KEY = \"your-api-key-here\"\n```\n\n" +
                "**For local development:** Create a `.streamlit/secrets.toml` file or `.env` file with:\n```\nGEMINI_API_KEY=your-api-key-here\n```")
        return
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'selected_query' not in st.session_state:
        st.session_state.selected_query = ''
    
    # Prepare data context
    context = {
        "total_customers": len(df),
        "avg_age": float(df['Age'].mean()),
        "age_range": f"{int(df['Age'].min())} to {int(df['Age'].max())}",
        "avg_income": float(df['Income'].mean()),
        "total_revenue": float(df['Total_Spent'].sum()),
        "avg_spending": float(df['Total_Spent'].mean()),
        "campaign_response_rate": float(df['Total_Campaigns_Accepted'].sum() / (len(df) * 6) * 100),
        "high_value_customers": int((df['Total_Spent'] > df['Total_Spent'].quantile(0.75)).sum()),
        "at_risk_customers": int((df['Recency'] > 90).sum()),
    }
    
    # Example questions
    st.markdown("#### 💡 Quick Questions:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 High-Value Customers", use_container_width=True):
            st.session_state.selected_query = "What are the key characteristics of my high-value customers?"
    
    with col2:
        if st.button("🎯 Reduce Churn", use_container_width=True):
            st.session_state.selected_query = "What strategies can I use to reduce customer churn?"
    
    with col3:
        if st.button("💰 Increase Revenue", use_container_width=True):
            st.session_state.selected_query = "What are the best strategies to increase customer revenue?"
    
    with col4:
        if st.button("📧 Target Segments", use_container_width=True):
            st.session_state.selected_query = "Which customer segments should I target for the next campaign?"
    
    st.markdown("---")
    
    # Chat input
    user_query = st.text_input(
        "💬 Your Question:",
        value=st.session_state.selected_query,
        placeholder="Ask anything about your customers...",
        key="chat_input_field"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("🚀 Ask AI", type="primary")
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.selected_query = ''
            st.rerun()
    
    # Process query
    if send_button and user_query:
        with st.spinner("🤔 AI is thinking..."):
            try:
                response = get_gemini_response(user_query, context, api_key)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": response
                })
                st.session_state.selected_query = ''
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📝 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <p style='color: white; margin: 0; font-size: 16px;'><b>🧑 You:</b> {chat['question']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 5px 15px; border-radius: 10px; margin: 5px 0 10px 0;'>
                <p style='color: white; margin: 0; font-size: 14px;'><b>🤖 AI Assistant</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(chat['answer'])
            
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")


def get_gemini_response(query, context, api_key):
    """Get response from Google Gemini API"""
    try:
        import requests
        
        system_prompt = f"""You are an expert customer analytics AI assistant analyzing this data:

**Customer Base:**
- Total: {context['total_customers']:,} customers
- Age: {context['avg_age']:.1f} years (range: {context['age_range']})
- Income: ${context['avg_income']:,.2f} average
- Revenue: ${context['total_revenue']:,.2f} total
- Spending: ${context['avg_spending']:,.2f} per customer
- Campaign Response: {context['campaign_response_rate']:.1f}%
- High-Value: {context['high_value_customers']} customers ({(context['high_value_customers']/context['total_customers']*100):.1f}%)
- At-Risk: {context['at_risk_customers']} customers ({(context['at_risk_customers']/context['total_customers']*100):.1f}%)

Provide detailed, data-driven insights with specific recommendations. Use markdown formatting (###, **, -) for clarity."""

        data = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt}\n\n**Question:** {query}\n\nProvide actionable insights:"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            }
        }
        
        # Use v1 API with gemini-2.5-flash model (latest available)
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key}",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "⚠️ No response from Gemini. Please try again."
        elif response.status_code == 400:
            error_detail = response.json()
            return f"⚠️ Invalid API request: {error_detail.get('error', {}).get('message', 'Unknown error')}"
        elif response.status_code == 403:
            return f"⚠️ API key permission denied. Please verify your key at https://aistudio.google.com/apikey"
        elif response.status_code == 429:
            return f"⚠️ Rate limit exceeded. Please wait a moment and try again."
        else:
            error_text = response.text[:300] if len(response.text) > 300 else response.text
            return f"⚠️ API Error ({response.status_code}): {error_text}"
            
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "⚠️ Connection error. Please check your internet connection."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


if __name__ == "__main__":
    main()
