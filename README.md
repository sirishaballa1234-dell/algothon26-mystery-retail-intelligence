# algothon26-mystery-retail-intelligence
Data Science solution
# 📊 Mystery Retail Intelligence

## ALGOTHON'26 — ALG-DATA-01: The Mystery Dataset

A data science project that explores the UCI Online Retail dataset to discover
hidden business patterns, customer behavior, anomalies, and revenue trends
without starting with a predefined business question.

---

## 🎯 Problem Statement

The Mystery Dataset problem provides a large dataset without a predefined
business question.

Our objective is to:

- Understand the dataset
- Clean and preprocess the data
- Discover meaningful patterns
- Analyze relationships between variables
- Detect unusual transactions
- Segment customers
- Formulate and test hypotheses
- Build a predictive model
- Present actionable business insights

---

## 💡 Solution

We developed a complete exploratory and analytical pipeline:

1. Dataset loading
2. Data quality analysis
3. Data cleaning
4. Exploratory Data Analysis
5. Revenue analysis
6. Product and country analysis
7. Relationship analysis
8. Customer RFM segmentation
9. Anomaly detection
10. Hypothesis testing
11. Revenue prediction
12. Business insights

---

## 📂 Dataset

Dataset used:

**UCI Online Retail Dataset**

The dataset contains transactions from a UK-based online retail business.

Important variables include:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

---

## 🧹 Data Cleaning

The following preprocessing steps were performed:

- Removed duplicate records
- Handled missing values
- Converted InvoiceDate to datetime
- Identified cancelled invoices
- Removed invalid quantities
- Removed invalid prices
- Created a new Revenue/Amount variable

Revenue is calculated as:

`Amount = Quantity × UnitPrice`

---

## 📊 Exploratory Data Analysis

The project analyzes:

- Total revenue
- Number of orders
- Number of customers
- Number of products
- Country-wise revenue
- Product-wise revenue
- Daily revenue trends

---

## 🔗 Relationship Analysis

Spearman correlation is used to investigate relationships between:

- Quantity
- Unit Price
- Revenue

A correlation heatmap is generated to visualize these relationships.

---

## 👥 Customer Segmentation

RFM analysis is used to classify customers based on:

- Recency
- Frequency
- Monetary value

Customers are grouped into segments such as:

- Champions
- Loyal / Active
- Potential
- At Risk
- Hibernating

---

## 🚨 Anomaly Detection

Isolation Forest is used to identify unusual transactions.

The model considers:

- Quantity
- Unit Price
- Transaction Amount

These anomalies represent statistically unusual transactions and are not
automatically considered fraudulent.

---

## 🔬 Hypothesis Analysis

The project investigates whether high-quantity purchases are associated
with different unit-price behavior.

Spearman correlation and quantity-group comparisons are used to investigate
the hypothesis.

---

## 🤖 Revenue Prediction

A Random Forest Regression model is used to predict daily revenue.

Features include:

- Previous-day revenue
- Previous-week revenue
- Rolling 7-day revenue
- Day of week
- Month

The data is split chronologically to avoid time-series leakage.

Evaluation metrics:

- MAE
- RMSE

---

## 🔎 Key Discoveries

The analysis is designed to uncover:

- High-value products
- Important markets
- Customer segments contributing significant revenue
- Unusual transaction patterns
- Revenue trends
- Relationships between transaction variables
- Potential business opportunities

The exact findings are generated from the dataset during notebook execution.

---

## 🏗️ Architecture

```text
                UCI Online Retail Dataset
                          │
                          ▼
                  Data Collection
                          │
                          ▼
                  Data Cleaning
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
             EDA      Relationship   Data
                      Analysis       Analysis
              │           │           │
              ▼           ▼           ▼
          Visualizations Hypotheses Anomaly Detection
                          │
                          ▼
                  Customer Segmentation
                          │
                          ▼
                    ML Prediction
                          │
                          ▼
                 Business Insights
