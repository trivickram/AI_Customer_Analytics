# 🎯 AI-Powered Customer Recommendation Dashboard

A comprehensive Streamlit dashboard implementing **4 AI-powered recommendation systems** for customer behavior analysis and marketing optimization.

![Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)

## 🚀 Features

This dashboard implements four powerful AI systems:

### 1. 🎯 Customer Segmentation (K-Means Clustering)
- **Type:** Unsupervised Learning
- **Purpose:** Automatically discover customer groups with shared characteristics
- **Algorithms:** K-Means Clustering with PCA visualization
- **Business Value:** 
  - Identify distinct customer personas (Premium Shoppers, Budget-Conscious, Family Focused, etc.)
  - Tailor marketing messages to each segment
  - Optimize product offerings per group
- **Output:** Customer segments with detailed personas and actionable recommendations

### 2. 📧 Campaign Response Prediction (Classification)
- **Type:** Supervised Learning
- **Purpose:** Predict probability of customer responding to marketing campaigns
- **Algorithms:** Random Forest, Logistic Regression, Gradient Boosting
- **Business Value:**
  - Create propensity-to-buy models
  - Target only high-probability customers (top 20%)
  - Drastically increase conversion rates and reduce marketing costs
  - Calculate ROI improvement over mass marketing
- **Output:** Propensity scores, confusion matrix, feature importance, ROI calculator

### 3. 🛒 Market Basket Analysis (Association Rules)
- **Type:** Unsupervised Learning
- **Purpose:** Discover products frequently purchased together
- **Algorithms:** Apriori Algorithm, Association Rules Mining
- **Business Value:**
  - Find cross-selling opportunities
  - Create product bundles
  - Optimize product placement in stores/websites
  - Generate "frequently bought together" recommendations
- **Output:** Association rules with support, confidence, and lift metrics

### 4. 💎 Customer Lifetime Value Prediction (Regression)
- **Type:** Supervised Learning
- **Purpose:** Predict total spending potential of each customer
- **Algorithms:** Gradient Boosting, Random Forest Regression
- **Business Value:**
  - Identify VIP customers requiring retention focus
  - Find characteristics of high-value customers
  - Segment customers by value (VIP, High, Medium, Low)
  - Optimize resource allocation based on customer value
- **Output:** CLV predictions, customer value segments, retention strategies

## 📊 Dataset

The dashboard uses the **Marketing Campaign Dataset** with the following information:

**Customer Demographics:**
- Age, Income, Education, Marital Status
- Number of children, household composition

**Purchase Behavior:**
- Spending on: Wines, Fruits, Meat, Fish, Sweets, Gold Products
- Purchase channels: Web, Catalog, Store
- Recency, Frequency, Monetary metrics

**Campaign Response:**
- Response to 5 previous campaigns
- Current campaign acceptance

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
```bash
cd d:\Customer_AI
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- mlxtend

### Step 3: Verify Data File
Make sure `marketing_campaign.csv` is in the same directory as `streamlit_dashboard.py`

## 🚀 Running the Dashboard

### Method 1: Command Line
```bash
streamlit run streamlit_dashboard.py
```

### Method 2: VS Code Terminal
1. Open terminal in VS Code
2. Navigate to the project folder
3. Run: `streamlit run streamlit_dashboard.py`

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## 📖 How to Use

### 1. Overview Tab
- View dataset statistics
- Understand the 4 AI systems
- Preview and download data

### 2. Customer Segmentation Tab
- Configure number of clusters (2-8)
- Click "Run Segmentation"
- View PCA visualization of segments
- Analyze segment characteristics
- Get personalized recommendations for each segment

### 3. Campaign Prediction Tab
- Select classification model
- Train the model
- View confusion matrix and performance
- Analyze feature importance
- Use ROI calculator to compare targeted vs. mass marketing

### 4. Market Basket Tab
- Set minimum support and confidence thresholds
- Find association rules
- View lift vs. confidence scatter plot
- Get cross-selling and bundling recommendations
- Identify product placement opportunities

### 5. CLV Prediction Tab
- Select regression model
- Train CLV predictor
- View actual vs. predicted scatter plot
- Analyze CLV distribution and segments
- Get retention strategies for each value segment

## 💡 Business Use Cases

### Marketing Teams
- **Segmentation:** Create targeted campaigns for different customer personas
- **Campaign Optimization:** Only contact customers likely to respond
- **ROI Improvement:** Reduce marketing costs by 60-80% while maintaining conversions

### Sales Teams
- **Cross-Selling:** Recommend complementary products based on purchase patterns
- **Upselling:** Identify high-value customers for premium offerings
- **Product Bundling:** Create bundles based on association rules

### Customer Success Teams
- **VIP Management:** Focus retention efforts on high-CLV customers
- **Churn Prevention:** Monitor at-risk high-value customers
- **Personalization:** Tailor experiences based on segment characteristics

### Product Teams
- **Product Development:** Understand preferences of different segments
- **Pricing Strategy:** Optimize pricing for value segments
- **Feature Prioritization:** Focus on features valued by high-CLV customers

## 📊 Key Metrics & Outputs

### Segmentation Metrics
- Silhouette Score (cluster quality)
- Segment size and distribution
- Average characteristics per segment

### Campaign Prediction Metrics
- Accuracy, Precision, Recall
- Confusion Matrix
- ROI comparison (targeted vs. mass marketing)
- Feature importance scores

### Market Basket Metrics
- Support (frequency of itemsets)
- Confidence (conditional probability)
- Lift (strength of association)
- Number of association rules

### CLV Prediction Metrics
- R² Score (prediction accuracy)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- CLV distribution and segments

## 🎨 Dashboard Features

- **Interactive Visualizations:** Plotly charts with hover details
- **Real-time Model Training:** Train models with custom parameters
- **Export Capabilities:** Download processed data and results
- **Responsive Design:** Works on desktop and tablet
- **Professional Styling:** Clean, modern interface
- **Session Management:** Retains model state across tabs

## 🔧 Customization

### Modify Clustering
Edit `customer_segmentation()` function:
- Change `clustering_features` list
- Adjust K-Means parameters
- Modify persona naming logic

### Add More Models
Edit `campaign_response_prediction()` or `clv_prediction()`:
- Import additional scikit-learn models
- Add to model selection dropdown
- Configure hyperparameters

### Customize Visualizations
All Plotly charts can be modified:
- Change color schemes: `color_discrete_sequence`
- Adjust chart types: `px.scatter`, `px.bar`, etc.
- Add annotations and highlights

## 📈 Expected Results

### Segmentation
- **4-6 distinct customer segments** with clear characteristics
- Personas like: Premium Shoppers, Budget Families, Senior Loyalists
- Silhouette scores typically 0.3-0.5

### Campaign Prediction
- **Accuracy: 85-90%** for response prediction
- **ROI improvement: 100-300%** vs. mass marketing
- Top features: Income, Previous campaign response, Recency

### Market Basket
- **10-50 association rules** depending on thresholds
- Common patterns: Wine + Meat, Gold + Wine
- Lift values: 1.5-3.0 for strong associations

### CLV Prediction
- **R² Score: 0.75-0.85** for spending prediction
- Clear separation between VIP and low-value customers
- Top predictors: Income, Purchase frequency, Age

## 🐛 Troubleshooting

### Issue: Module not found
**Solution:** Install missing package
```bash
pip install <package-name>
```

### Issue: CSV file not found
**Solution:** Ensure `marketing_campaign.csv` is in the same directory
```bash
dir  # Windows
ls   # Mac/Linux
```

### Issue: Port already in use
**Solution:** Specify different port
```bash
streamlit run streamlit_dashboard.py --server.port 8502
```

### Issue: Dashboard not loading
**Solution:** Clear cache and restart
```bash
streamlit cache clear
streamlit run streamlit_dashboard.py
```

## 📚 Resources

### Machine Learning
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [K-Means Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Association Rules Mining](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

### Business Applications
- Customer Segmentation Best Practices
- Marketing ROI Optimization
- CLV Calculation Methods

## 🤝 Contributing

Feel free to enhance this dashboard:
1. Add new AI systems (e.g., Churn Prediction, Recommendation Engine)
2. Implement A/B testing framework
3. Add real-time data integration
4. Create automated reporting features
5. Build mobile-responsive views

## 📄 License

This project is created for educational and business analysis purposes.

## 👨‍💻 Author

Created with ❤️ for data-driven marketing excellence.

---

**🎯 Ready to transform your marketing with AI?**

Run the dashboard and start discovering insights:
```bash
streamlit run streamlit_dashboard.py
```
