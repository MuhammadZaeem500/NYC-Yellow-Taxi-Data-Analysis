# 🚕 NYC Yellow Taxi Trip Analysis & Prediction

An end-to-end data science and full-stack machine learning web application analyzing approximately **2.3 million NYC Yellow Taxi trip records** (Jan–Mar 2023) provided by the NYC Taxi & Limousine Commission (TLC). 

[![Live Demo](https://img.shields.io/badge/Live-Render-purple?style=for-the-badge&logo=render)](https://nyc-yellow-taxi-data-analysis.onrender.com/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/MuhammadZaeem500/NYC-Yellow-Taxi-Data-Analysis)

---

## 🚀 Live Application Preview

| Main Interface | Cash Payment Prediction | Credit Card Payment Prediction |
| :---: | :---: | :---: |
| ![Application UI](Application%20UI.jpg) | ![Cash Payment](Cash%20Payment.jpg) | ![Credit Card Payment](Credit%20Card%20Payment.jpg) |

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Technologies Used](#3-technologies-used)
4. [Data Cleaning](#4-data-cleaning)
5. [Exploratory Data Analysis & Visualizations](#5-exploratory-data-analysis--visualizations)
6. [Machine Learning Models](#6-machine-learning-models)
7. [Results & Model Performance](#7-results--model-performance)
8. [Skills Demonstrated](#8-skills-demonstrated)
9. [Deployment & Links](#9-deployment--links)

---

## 1. Project Overview

This project focuses on analyzing NYC Yellow Taxi trip data using Python and various data analysis and machine learning techniques. The dataset contains approximately 2.3 million taxi trip records with information such as trip distance, fare amount, passenger count, pickup and drop-off locations, payment type, and trip time. The main goal of the project was to clean and prepare the data, explore useful patterns and trends, create visualizations, and build machine learning models for regression and classification, wrapped inside a polished, production-ready web application.

---

## 2. Dataset

The project uses NYC Yellow Taxi trip data provided in Parquet format via the official NYC TLC trip record data portal. 

* **Size:** ~2.3 million rows | 19 columns
* **Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
* **Key Attributes Included:**
  * Pickup and drop-off dates and times
  * Passenger count and trip distance
  * Fare amount, tip amount, tolls amount, and total amount
  * Payment type and rate code IDs
  * Pickup and drop-off location coordinates

---

## 3. Technologies Used

* **Programming Language:** Python
* **Data Processing & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (`LinearRegression`, `LogisticRegression`)
* **Backend Framework:** FastAPI, Uvicorn, Joblib, Pydantic
* **Frontend Design:** Tailwind CSS v4, HTML5, JavaScript (Fetch API)
* **Development Environment:** Jupyter Notebook, Visual Studio Code, Git
* **Data Format:** Parquet

---

## 4. Data Cleaning

The raw dataset underwent rigorous data cleansing and feature engineering before feeding into machine learning pipelines:
* Checked structure, data types, and identified missing records.
* Handled invalid entries by filtering out unrealistic values (e.g., negative or zero fares/distances, fares exceeding \$500, or distances over 100 miles).
* Restricted passenger counts to valid capacities (1 to 6 passengers) and filtered valid Ratecode IDs (1–6).
* Extracted temporal and spatial features such as `pickup_hour`, `pickup_day`, `pickup_month`, and calculated `trip_duration` in minutes.

### Data Distribution Comparison
| Before Cleaning — Raw Data | After Cleaning — Clean Data |
| :---: | :---: |
| ![EDA Before Cleaning Distributions](EDA%20Before%20Cleaning%20Distributions.png) | ![EDA After Cleaning Distributions](EDA%20After%20Cleaning%20Distributions.png) |

---

## 5. Exploratory Data Analysis & Visualizations

Exploratory Data Analysis (EDA) was performed to uncover key patterns and behavioral trends across New York City taxi trips:

* **Trips by Hour of Day:** Peak demand occurs during evening rush hours (5 PM – 6 PM), while volume bottoms out in the early morning hours (3 AM – 5 AM).
* **Payment Methods:** Credit card transactions dominate the market, accounting for **81.4%** of trips, followed by cash at **17.7%**.
* **Trip Distance vs. Fare:** Strong positive linear correlation demonstrating that longer trips yield higher fares.
* **Rate Codes:** Airport, Newark, and Nassau/West trips carry significantly higher average total amounts compared to standard city fares.

### Key Analytical Visualizations
| Number of Trips by Hour | Payment Method Distribution |
| :---: | :---: |
| ![Trips By Hour Of Day](Trips%20By%20Hour%20Of%20Day.png) | ![Payment Method Distribution](Payment%20Method%20Distribution.png) |

| Trip Distance vs Fare Amount | Average Fare by Passenger Count |
| :---: | :---: |
| ![Trip Distance Vs Fare Amount](Trip%20Distance%20Vs%20Fare%20Amount.png) | ![Average Fare By Passenger Count](Average%20Fare%20By%20Passenger%20Count.png) |

| Average Trip Duration by Hour | Average Total Amount by Rate Code |
| :---: | :---: |
| ![Average Trip Duration By Hour](Average%20Trip%20Duration%20By%20Hour.png) | ![Average Total Amount By Rate Code](Average%20Total%20Amount%20By%20Rate%20Code.png) |

---

## 6. Machine Learning Models

Two independent machine learning models were serialized using `joblib` and integrated directly into the production FastAPI backend:

1. **Model 1 — Fare Amount Predictor (Regression):**
   * **Algorithm:** Linear Regression
   * **Features:** `trip_distance`, `trip_duration`, `pickup_hour`, `passenger_count`, `RatecodeID`
   * **Target:** `fare_amount`

2. **Model 2 — Payment Type Classifier (Classification):**
   * **Algorithm:** Logistic Regression (with `class_weight='balanced'`)
   * **Features:** `trip_distance`, `trip_duration`, `pickup_hour`, `fare_amount` (predicted), `passenger_count`
   * **Target:** `payment_type` (Credit Card vs. Cash)

---

## 7. Results & Model Performance

* **Linear Regression (Fare Prediction):**
  * **MAE:** \$1.51
  * **RMSE:** \$3.94
  * **$R^2$ Score:** **0.946 (94.6% variance explained)**
  * *Verdict:* Exceptional performance. Proves that trip distance and duration heavily dictate fare structures.

* **Logistic Regression (Payment Classification):**
  * **Accuracy:** 56.77%
  * *Verdict:* Highlights the challenge of predicting payment choice using trip features alone, as passenger payment behavior spans across demographics regardless of simple trip metrics.

---

## 8. Skills Demonstrated

* Python Programming & Advanced Data Manipulation
* Data Cleaning & Preprocessing at Scale (~2.3M rows)
* Exploratory Data Analysis & Statistical Interpretation
* Custom Data Visualization (Matplotlib & Seaborn)
* Feature Engineering & Pipeline Management
* Supervised Machine Learning (Regression & Classification)
* Full-Stack Web App Development (FastAPI, Tailwind CSS, JavaScript)
* Cloud Deployment & Model Serialization (`Joblib`, `Render`)

---

## 9. Deployment & Links

* **Live Application:** [https://nyc-yellow-taxi-data-analysis.onrender.com/](https://nyc-yellow-taxi-data-analysis.onrender.com/)
* **GitHub Repository:** [https://github.com/MuhammadZaeem500/NYC-Yellow-Taxi-Data-Analysis](https://github.com/MuhammadZaeem500/NYC-Yellow-Taxi-Data-Analysis)