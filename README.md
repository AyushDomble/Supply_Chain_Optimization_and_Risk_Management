# 📦 Supply Chain Optimization & Risk Management Dashboard

A comprehensive **data-driven dashboard** built with **Python** and **Streamlit** to analyze supply chain performance, predict late delivery risks, and provide actionable insights for logistics management.

---

## 🚀 Live Demo  
[Add your Streamlit Community Cloud deployment link here]

---

## ✨ Key Features
- **Executive KPIs**: Quick insights with OTIF (On-Time-In-Full) Rate & Perfect Order Rate.  
- **Interactive Analytics**: Dynamic charts & geographical heatmaps to spot high-risk regions.  
- **ML-Powered Risk Prediction**: Real-time late delivery risk detection with Random Forest.  
- **Optimal Shipping Recommendation**: Suggests best shipping mode balancing risk & profit.  
- **Dynamic Filtering & Export**: Filter by region, mode, date & export results as CSV.  

---

## 📸 Dashboard Preview  
   
(Add a screenshot of your running dashboard here)

---

## 📂 Project Structure
```plaintext
supply_chain_optimization/
├── 📂 dashboard/
│   └── 📄 app.py              # Main Streamlit application
├── 📂 data/
│   ├── 📂 raw/
│   │   └── 📄 DataCoSupplyChainDataset.csv
│   └── 📂 processed/
│       ├── 📄 cleaned_orders.csv
│       ├── 📄 customer_segments.csv
│       ├── 📄 features.csv
│       └── 📄 final_data_with_segments.csv
├── 📂 models/
│   ├── 📄 customer_segmenter.joblib
│   ├── 📄 delivery_risk_pipeline.joblib
│   ├── 📄 demand_forecaster_Europe.pkl
│   ├── 📄 demand_forecaster_LATAM.pkl
│   └── 📄 ... (and other regional models)
├── 📂 notebooks/
│   ├── 📄 1.0-data_exploration_and_cleaning.ipynb
│   ├── 📄 2.0-feature_engineering.ipynb
│   ├── 📄 3.0-descriptive_analytics.ipynb
│   ├── 📄 4.0-ml_delivery_risk_prediction.ipynb
│   ├── 📄 5.0-ml_customer_segmentation.ipynb
│   ├── 📄 6.0-ml_demand_forecasting.ipynb
│   └── 📄 7.0-optimization_models.ipynb
├── 📂 src/
│   ├── 📂 data/
│   │   └── 📄 make_dataset.py
│   ├── 📂 features/
│   │   ├── 📄 add_segments.py
│   │   └── 📄 build_features.py
│   └── 📂 models/
│       └── 📄 train_model.py
├── 📄 .gitignore
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 run_pipeline.py
```
---

## 🛠 Tech Stack
- **Backend**: Python, Pandas, Scikit-learn  
- **Machine Learning**: Random Forest (classification), K-Means (segmentation), Prophet (forecasting)  
- **Dashboard**: Streamlit  
- **Visualization**: Plotly Express  

---

## ⚙️ How to Run Locally

# Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
# Create & activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
# Install dependencies
```bash
pip install -r requirements.txt
```
# Run data pipeline (process data + train ML models)
```bash
python run_pipeline.py
```
# Launch dashboard
```bash
streamlit run dashboard/app.py
```

## 📊 Data Source
This project uses the **[DataCo Global Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)** dataset, which is publicly available on Kaggle.

---

## 📄 License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

