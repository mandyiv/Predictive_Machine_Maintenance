**PROJECT: Predictive Maintenance System for Industrial Machines**

**Project Overview:**
Developed and deployed a predictive maintenance system using the AI4I 2020 Predictive Maintenance Dataset. The project involved building an end-to-end machine learning pipeline to predict machine failures, ensuring timely maintenance and reducing downtime.

**Key Responsibilities:**

**Data Preprocessing and Feature Engineering:**
- Conducted exploratory data analysis to understand the dataset and identify key features.
- Handled missing values and performed data cleaning to prepare the dataset for modeling.
- Converted categorical variables to numerical values using one-hot encoding.
- Scaled continuous features using StandardScaler to normalize the data.
- Addressed class imbalance using Synthetic Minority Over-sampling Technique (SMOTE) to improve model performance.

**Model Training and Evaluation:**
- Split the dataset into training and test sets to evaluate model performance.
- Trained a Random Forest Classifier to predict machine failures, leveraging its robustness and interpretability.
- Evaluated the model using metrics such as accuracy, precision, recall, and F1-score to ensure reliable predictions.

**Model Deployment:**
- Implemented a Flask web application to serve the trained model as a web service.
- Designed a user-friendly HTML interface for inputting machine data and receiving predictions.
- Deployed the Flask application, enabling real-time predictions for machine maintenance.

**Tools and Technologies:**
- **Programming Languages:** Python
- **Libraries and Frameworks:** scikit-learn, pandas, streamlit, imbalanced-learn, joblib
- **Development Tools:** Jupyter Notebook, Git, Virtualenv
- **Data Processing:** One-hot encoding, StandardScaler, SMOTE
- **Modeling Techniques:** Random Forest Classifier, Train-Test Split, Model Evaluation Metrics

**Project Impact:**
- Successfully reduced the downtime of industrial machines by predicting potential failures in advance.
- Improved maintenance scheduling and resource allocation, leading to cost savings and enhanced operational efficiency.
