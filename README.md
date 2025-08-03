# Shopper Spectrum

## Customer Segmentation & Product Recommendation System

---

### Project Overview

**Shopper Spectrum** is a data-driven solution designed to analyze retail transaction data for customer segmentation and product recommendation. Using machine learning techniques, this project segments customers based on their purchasing behavior and provides personalized product recommendations to enhance marketing strategies and customer engagement.

The solution includes:
- Customer segmentation via RFM (Recency, Frequency, Monetary) analysis followed by K-Means clustering.
- Collaborative filtering-based product recommendation system using K-Nearest Neighbors (KNN).
- An interactive **Streamlit** web app that enables users to explore the data, predict customer segments, and get product recommendations with rich visualizations.

---

### Features

- **Interactive Dashboard:**
  - Displays key metrics such as total customers, products, transactions, and revenue.
  - Visualizes transaction volumes by country and top-selling products.
  - Shows purchase trends over time.

- **Customer Segmentation Module:**
  - Users input Recency, Frequency, and Monetary values to predict customer segment.
  - Segments reflect real-world retail customer groups (e.g., High-Value, Loyal, At-Risk).
  - 3D scatter plot visualization of customer segments.

- **Product Recommendation Module:**
  - Users select a product to receive the top 5 similar product recommendations.
  - Displays similarity scores and product information.
  - Interactive product similarity heatmap.

---

### Technologies & Libraries

- **Python 3.11**
- Data Handling: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Machine Learning: `scikit-learn`
- Web App: `streamlit`
- Others: `pickle` (model serialization), `scipy` (matrix operations)

---

### Methodology

1. **Data Preprocessing:**
   - Clean and filter raw transaction data (`online_retail.csv`).
   - Generate RFM metrics per customer.
   
2. **Customer Segmentation:**
   - Transform and standardize RFM features.
   - Use K-Means clustering for segment discovery.
   - Interpret and label clusters into meaningful customer types.

3. **Product Recommendation:**
   - Construct customer-product purchase matrix.
   - Train KNN model on product similarities via purchase patterns.
   - Recommend related products based on item similarity.

4. **Model Saving and Deployment:**
   - Models and data transformers saved via `pickle`.
   - Streamlit app loads models for real-time inference and visualization.

---

### Installation & Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/shopper-spectrum.git
    cd shopper-spectrum
    ```

2. **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Ensure Dataset and Models are in Place:**

    - Place `online_retail.csv` dataset in the project root.
    - Pretrained models (`kmeans_model.pkl`, `scaler.pkl`, `recommendation_model.pkl`, `cluster_names.pkl`) should be located in the project directory as well.
    - If missing, please run the analysis notebook or script (`customer_segmentation.ipynb`) to generate models.

4. **Run the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

5. **Access the App:**

    Open your browser and navigate to `http://localhost:8501`.

---

### Project Structure
```
project/ ├── online_retail.csv # Retail transaction dataset ├── customer_segmentation.ipynb # Data analysis and model building notebook ├── app.py # Streamlit web application ├── requirements.txt # Python dependencies ├── kmeans_model.pkl # Saved KMeans customer segmentation model ├── scaler.pkl # Saved scaler for feature standardization ├── recommendation_model.pkl # Saved KNN recommendation model & data ├── cluster_names.pkl # Dictionary for cluster label mappings └── README.md # Project documentation
```

---

### How to Use

- **Dashboard:** Explore insights on customer base, product sales, and purchase trends.
- **Customer Segmentation:** Input RFM values to classify customers into meaningful segments; useful for targeting and marketing strategies.
- **Product Recommendations:** Select any product to discover similar products to promote or cross-sell.

---

### Contribution

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

### References & Further Reading

- [RFM Analysis](https://en.wikipedia.org/wiki/RFM_(customer_value))
- [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

### License

This project is released under the MIT License.

---

### Contact

For questions or feedback, please reach out to the project maintainer.

---

Thank you for using Shopper Spectrum! 🔍🛍️