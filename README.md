# 📊 Customer Segmentation Project - KMeans

🚀 Overview
 ---

This project aims to segment customers based on demographic and behavioral data using K-Means Clustering. The goal is to help businesses better understand customer groups and apply targeted marketing strategies.

 ---
 
📌 Task Description
---
An automobile company plans to expand into new markets using existing products (P1–P5). Based on previous success, customers were segmented into 4 groups (A, B, C, D) and targeted with personalized strategies.

🎯 Objective:

Predict the appropriate customer segment for new potential customers (2,627 records) using unsupervised learning.

 ---
 
📂 Dataset Information
---
The dataset contains customer demographic and behavioral features such as:

- Gender
- Age
- Ever Married
- Graduated
- Profession
- Work Experience
- Spending Score
- Family Size
- Category (Var_1)

📊 Includes:

- Training data (labeled with segments A–D)
- Test data (unlabeled new customers)

  ---
  
🛠️ Tools & Technologies
---

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit (for deployment)
- Joblib (model saving)

  ---
  
🔄 Project Workflow
---

- 1. Data Overview & Cleaning

         - Checked missing values and data types
         - Handled categorical variables
         - Applied binary encoding (Yes/No, Gender)
     
- 2. Exploratory Data Analysis (EDA)

         - Distribution of age, income, and spending
         - Customer behavior patterns
         - Feature relationships across segments
     
      <img width="989" height="574" alt="image" src="https://github.com/user-attachments/assets/c3475bb6-c97e-450c-8434-0d456c6e9278" />

- 3. Feature Engineering

         - Applied frequency encoding for:
         - Profession
         - Category (Var_1)
         - Log transformation (log1p) for stability
     
- 4. Preprocessing
     
         - Standardization using StandardScaler
         - One-Hot Encoding for Spending Score
         - Combined using ColumnTransformer

- 5. Modeling (K-Means Clustering)

         - Applied K-Means clustering
         - Tested multiple values of k

- 6. Model Evaluation
       
         - 📉 Elbow Method → Optimal number of clusters
         - 📈 Silhouette Score → Cluster quality
         - 👉 Optimal clusters selected: k = 4
     
     <img width="589" height="455" alt="image" src="https://github.com/user-attachments/assets/e153e6d1-dde6-4090-998f-1b0437ae1e3d" />

 ---

📊 Model Results
---

 <img width="689" height="547" alt="image" src="https://github.com/user-attachments/assets/3cbe238a-00f7-4d09-9088-8daf15327388" />

- Successfully segmented customers into 4 distinct groups
- Balanced distribution across clusters:

      - Cluster 0 → 2470 customers
      - Cluster 1 → 1952 customers
      - Cluster 2 → 2153 customers
      - Cluster 3 → 1493 customers
- Model integrated into interactive Streamlit app
  ---
  
💡 Business Insights
---

 <img width="678" height="547" alt="image" src="https://github.com/user-attachments/assets/b9ceffb1-5674-43bf-b58a-a9674ff9cf57" />

- 👨‍👩‍👧 Budget Family Customers
  - → Large families, low spending → Target with discounts & bundles

- 💼 Premium Potential Customers
  - → Balanced spending → Ideal for loyalty & premium offers

- 🌱 Emerging Customers
  - → Younger, growing spending → Good for upselling

- 🏆 Established Professionals
  - → Stable income → High-value product targeting

 ---
 
🧠 Concepts Covered
---
- Data Cleaning & Preprocessing
- Feature Engineering (Frequency Encoding)
- Scaling & Encoding Techniques
- Unsupervised Learning (K-Means)
- Model Evaluation (Elbow & Silhouette)
- Data Visualization (EDA & Dashboard)
- Model Deployment (Streamlit)

 ---
 
📌 Application
---

An interactive dashboard was built using Streamlit to:

- Predict customer segment
- Visualize customer profile
- Display insights and KPIs

 ---
 
👩‍💻 Author
---

Atikah Dwi Rizky

📊 Data Science & Machine Learning Enthusiast
