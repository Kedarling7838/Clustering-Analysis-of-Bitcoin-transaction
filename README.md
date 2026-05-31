# Clustering Analysis of Bitcoin Transactions for Entity Behavior Profiling

## Project Overview

This project applies clustering techniques to Bitcoin transaction data to identify and profile behavioral patterns among different cryptocurrency entities. By analyzing transaction characteristics such as transaction value, transaction fees, sender connections, and receiver connections, the project groups similar entities together and uncovers hidden structures within the Bitcoin ecosystem.

The goal is to support financial analysis, compliance monitoring, fraud detection, and blockchain intelligence through unsupervised machine learning.

---

## Business Problem

The Bitcoin network processes millions of transactions involving various entities such as:

- Exchanges
- Miners
- Traders
- Hodlers
- Mixers

Due to Bitcoin's pseudonymous nature, it is challenging to distinguish between different types of users and identify suspicious activity.

By clustering entities based on transaction behavior, organizations can:

- Identify behavioral groups
- Detect unusual transaction patterns
- Improve compliance and risk monitoring
- Gain insights into blockchain activity

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

## Dataset Description

The dataset contains synthetic Bitcoin transaction records with the following features:

| Feature | Description |
|----------|-------------|
| tx_id | Transaction ID |
| address | Wallet Address |
| entity_type | Entity Category |
| value_btc | Transaction Value in BTC |
| fee_btc | Transaction Fee |
| in_degree | Number of Unique Senders |
| out_degree | Number of Unique Receivers |
| timestamp | Transaction Time |

### Entity Types

- Exchange
- Miner
- Trader
- Hodler
- Mixer

---

## Project Workflow

### 1. Data Collection

- Loaded Bitcoin transaction dataset
- Examined dataset structure and features

### 2. Exploratory Data Analysis

Performed:

- Data type inspection
- Statistical summary analysis
- Distribution analysis
- Outlier detection

Key observations:

- Dataset contains 3000 transaction records
- No missing values found
- No duplicate records found
- Transaction value (`value_btc`) contains outliers

---

### 3. Data Cleaning

Performed:

- Missing value verification
- Duplicate record verification
- Feature selection

Removed non-essential attributes such as:

- Transaction IDs
- Wallet addresses
- Timestamp fields

for clustering purposes.

---

### 4. Feature Engineering

Selected clustering features:

```text
value_btc
fee_btc
in_degree
out_degree
```

Prepared data for clustering analysis.

---

### 5. Data Scaling

Applied feature scaling to normalize numerical values and improve clustering performance.

---

### 6. Clustering Model

Implemented clustering algorithms to group Bitcoin entities with similar transactional characteristics.

Possible behavioral clusters identified:

- High-volume traders
- Long-term holders
- Mining entities
- Cryptocurrency exchanges
- Mixing services

---

### 7. Cluster Analysis

Analyzed each cluster based on:

- Transaction volume
- Network connectivity
- Transaction fees
- Transaction frequency

Generated behavioral profiles for each cluster.

---

## Project Architecture

```text
Bitcoin Transaction Dataset
              │
              ▼
      Data Preprocessing
              │
              ▼
      Feature Selection
              │
              ▼
        Data Scaling
              │
              ▼
      Clustering Model
              │
              ▼
      Cluster Formation
              │
              ▼
    Behavioral Profiling
              │
              ▼
      Business Insights
```

---

## Project Structure

```text
Bitcoin-Clustering-Analysis/
│
├── Clustering Analysis of Bitcoin Transactions.py
├── btc_transactions_sample.csv
├── README.md
│
└── screenshots/
    ├── dataset_overview.png
    ├── clustering_output.png
    ├── cluster_visualization.png
    └── results.png
```

---

## Results

- Successfully grouped Bitcoin transaction entities based on behavioral patterns.
- Identified clusters representing different types of cryptocurrency participants.
- Detected transaction behavior similarities among entities.
- Generated meaningful insights from blockchain transaction data.

---

## Applications

- Blockchain Analytics
- Fraud Detection
- Cryptocurrency Compliance Monitoring
- Anti-Money Laundering (AML)
- Risk Assessment
- Financial Intelligence
- Entity Behavior Profiling

---

## Future Enhancements

- DBSCAN Clustering
- Hierarchical Clustering
- Real Blockchain Data Integration
- Anomaly Detection Models
- Interactive Dashboard
- Real-Time Blockchain Monitoring System

---

## Learning Outcomes

- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- Unsupervised Machine Learning
- Clustering Techniques
- Blockchain Data Analytics
- Cryptocurrency Transaction Analysis
- Behavioral Profiling

---

## Author

**Kedarling Ashok Kanade**

Bachelor of Engineering (Computer Science & Engineering)  
GM University, Davangere, Karnataka, India

### Connect with Me

- GitHub: https://github.com/your-github-username
- LinkedIn: https://www.linkedin.com/in/your-linkedin-profile

---

## License

This project is developed for academic and educational purposes. Feel free to use and modify it for learning and research activities.
