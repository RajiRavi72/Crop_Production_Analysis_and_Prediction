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
Project Report
Problem Statement:
The project is based on Agriculture domain. The project aims to develop a regression model that forecasts crop production (in tons) based on agricultural factors such as area harvested (in hectares), yield (in kg/ha), and the year, for various crops grown in a specific region.

Dataset:
Dataset: FAOSTAT_data
The FAOSTAT excel workbook consists of 2,24,648 rows of data pertaining to crops and livestock production in different countries in different years. 

Dataset Loading:
	The  Area_Harvested , Yield and Production are given in the excel file as 3 different rows with value column giving the values of the Area_Harvested, Yield and Production with units given in Unit column for the same values of other columns like the country, year, item etc. So, the three rows need to be transformed into 3 columns of the same row with their corresponding values. 
So after loading the dataset into pandas dataframe, the transformation is done by pivoting the data to wide format. Also the new columns are renamed as 'Area_Harvested', 'Yield_kg_per_ha', 'Production_tons'

Dataset Cleaning and Preprocessing:

1.	Removed the livestock data:
•	This is done by filtering only the crop records where ‘Area_Harvested’ is not null. 
•	Also irrelevant columns pertaining to livestock data is removed by filtering out only the required columns like 'Area', 'Item', 'Year', 'Area_Harvested', 'Yield_kg_per_ha', 'Production_tons'

2.	Check for Nulls:
•	Since our aim to create a ML model to predict the Crop Production, data in columns ‘Area_Harvested', 'Yield_kg_per_ha', 'Production_tons’ are very important. So we first remove the rows where there are nulls in these 3 columns.

3.	Checked for Unique values:
•	 Found  200 unique area pertaining to data from 200 different countries. 
•	Found 157 unique items pertaining to 157 different crops grown.
•	Found 5 unique values in Year pertaining to data for 5 years.

5.	Data Format: 
One hot encoding is done for Area and Item column to convert the categorical column into numerical format that machine learning models can understand.

Machine Learning Model:
1.	Features dataset (x) is created by dropping ‘Production_tons’ column. Target dataset (y) is created using only ‘Production_tons’ column.
2.	Created Training and Test dataset by splitting the X and y datasets  for Training (80 %) and for Testing (20%) by importing train_test_split from sklearn.model_selection  . Now we have 
X_train, X_test, y_train, y_test
3.	The following ML Models are then created:
  	1.	Linear Regression
  	2.	Random Forest Regressor
  	3.	Gradient Boosting Regressor
  	4.	K Neighbours Regressor
  	5.	Decision Tree Regressor
4.	Evaluation metrics are then used to evaluate the models. The metrics used are : 
  	1.	R2 Score
  	2.	MSE (Mean Square Error)
  	3.	MAE (Mean Absolute Error)
  	4.	MAPE (Mean Absolute Percentage Error)
      
Observations:
	It is found that Random Forest is the best suited model because of the following reasons:
    		1.	It has the highest R² (~0.9982), meaning it explains almost all the variance.
    		2.	It has the lowest MSE and MAE — much better than Linear Regression and Gradient Boosting.
    		3.	Its MAPE is very low (~1.79%), showing good prediction accuracy relative to actual values.
    	
Model Stored:
The Trained Random Forest model is stored in rf_model_crop_production.pkl  file.

Data Stored: 
	     The cleaned Dataset is stored as crop_data table in MSQL Workbench. 
      
Data Visualization: 

Streamlit Application is used for visualizing the data to get various insights.  

•	Crop Production Dashboard: Provides the total number of records, Number of unique crops and Unique Areas with Production in million tons, Average_Yield in kg/ha, and Year Range. Also shows the tops 10 crop Distribution as a Pie Chart.

•	Trend Analysis: Provides interactive analysis to know the Production over time for a particular crop grown in a particular Area.

•	Crop Distribution: Provides bar charts for Top 10 Most Cultivated Crops, Top 10 Least Cultivated Crops, Top 10 Regions by Crop Production and an interactive chart for the Most Produced crop in a selected Area.

•	Temporal Analysis: Line Charts to Visualize:
Area Harvested (in hectares) Over Time
Average Yield (in kg per ha) over Time
Crop Growth Analysis with Production (in tons) over Time
Growth Analysis: (Production Trend and Yield Trend) Crop-wise or Region-wise Trends
 
•	Environmental Relationships: 
A Scatterplot showing Area Harvested vs Yield is used to visualize the relationship and useful insights are drawn and Actionable recommendations provided.

•	Input-Output Relationships: 
   Correlation Heat Map: Provides insights of correlation between Area_Harvested, Yield_kg_per_ha and Production_tons. 

•	Comparative Analysis: 
Average Yield per Crop: Compares yields of different crops to identify high-yield vs. low-yield crops. 
Top Producing Regions: Compares Production across different Areas to find highly productive regions.

•	Outliers and Anomalies: 
Analysed using Yield Boxplot and Production Boxplot and various insights are drawn from the Boxplots.

•	Actionable Insights: 
1.	Focus on High Yeild Crops
2.	Regions with High Production
3.	Under utilized Crops (Low area harvested but High yields)


Production Prediction:
	Area dropdown provided to select the required Area where Production is being predicted.
	Crop dropdown provided to select the crop of interest.
	Year dropdown provided to select the year of production
	Area harvested in hectares is an input for which the prediction is made.
	Yield in kg per ha is again an input for which the prediction is made.
For the above values the prediction is made in tons using ML Algorithm.



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
