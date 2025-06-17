# Crop Production Analysis and Prediction

## 📊 Project Overview

This project is developed as part of the GUVI Capstone Project, focusing on analyzing agricultural crop production data and predicting future crop production using machine learning models. The data-driven analysis helps understand production patterns and forecast future trends based on area harvested, yield, and year.

---

## 🚀 Features

- Interactive Streamlit Web Application
- Data Cleaning and Preprocessing with SQL Workbench
- Visual Analytics using Seaborn and Matplotlib
- Regression Model for Production Prediction (Random Forest Regressor)
- SQL Database Integration
- Multiple Analytical Dashboards:
  - Crop Distribution
  - Temporal Analysis
  - Environmental Relationships
  - Input-Output Relationships
  - Comparative Analysis
  - Outliers & Anomalies

---

## 🗂️ Project Structure

Crop_Production_Prediction_Project/
│

├── data/ # Excel files (ignored from git)

├── database/ # SQL Scripts for table creation

├── notebook/ # Jupyter Notebooks for EDA & Model Building

├──notebook/ # Streamlit Application Files

├── env/ # Virtual Environment (ignored from git)

├── requirements.txt # Python dependencies

├── .gitignore

└── README.md # Project Documentation


---

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit**
- **scikit-learn**
- **Pandas**
- **Seaborn**
- **SQL Workbench**


---

## 🔧 Setup Instructions

### 1️⃣ Clone the Repository

git clone https://github.com/RajiRavi72/Crop_Production_Analysis_and_Prediction.git

2️⃣ Setup Virtual Environment (recommended)

cd Crop_Production_Analysis_and_Prediction

python -m venv env

source env/bin/activate   # For Linux/Mac

env\Scripts\activate      # For Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run Streamlit App

cd notebook

streamlit run app.py

📂 Data Source

The data used for this project comes from official agricultural production datasets provided during GUVI Capstone Project.

🙏 Acknowledgments

Special thanks to the GUVI AI & ML Program team for guidance and resources.

👩‍💻 Author

Raji Ravi - GitHub Profile
