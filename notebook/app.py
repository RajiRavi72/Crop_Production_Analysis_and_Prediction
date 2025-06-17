import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
import matplotlib.ticker as ticker


# DB Connection
engine = create_engine("mysql+pymysql://root:Raji@localhost:3306/crop_prediction_db")

# Load data
df = pd.read_sql("SELECT * FROM crop_data", con=engine)

# Load trained model
model = joblib.load('rf_model_crop_production.pkl')

# Sidebar Navigation
st.sidebar.title("🌾 Crop Production Analysis and Predictions")
page = st.sidebar.radio("Go to", (
    "Dashboard",
    "Trend Analysis",
    "Predict Production",
    "Crop Distribution",
    "Temporal Analysis",
    "Environmental Relationships",
    "Input-Output Relationships",
    "Comparative Analysis",
    "Outliers and Anomalies",
    "Actionable Insights"
     ))

# Dashboard
if page == "Dashboard":
    st.title("📊 Crop Production Dashboard")

    # Calculate metrics
    total_records = len(df)
    unique_crops = df['Item'].nunique()
    unique_areas = df['Area'].nunique()
    total_production = df['Production_tons'].sum() / 1_000_000  # in million tons
    average_yield = df['Yield_kg_per_ha'].mean()
    min_year = df['Year'].min()
    max_year = df['Year'].max()

    # Display main metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Unique Crops", unique_crops)
    col3.metric("Unique Areas", unique_areas)
    col4.metric("Production(Million Tons)", f"{total_production:,.2f}")

    col5, col6 = st.columns(2)
    col5.metric("Average Yield (kg/ha)", f"{average_yield:,.2f}")
    col6.metric("Year Range", f"{min_year} - {max_year}")

    st.markdown("---")

    # Add Top 10 Crops Distribution Pie Chart
    st.subheader("🌾 Top 10 Crops Distribution")
    top_crops = df.groupby('Item')['Production_tons'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    ax.pie(top_crops, labels=top_crops.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Top 10 Areas by Record Count
    st.header("🌎 Top 10 Areas by Record Count")
    df['Area'] = df['Area'].replace({
    'T�rkiye': 'Türkiye'   
})
    top_areas = df['Area'].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_areas.index, y=top_areas.values, ax=ax2)
    ax2.set_ylabel("Record Count")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig2)

    st.markdown("---")

    # Insights Box
    st.subheader("📦 Dataset Insights")
    st.info("""
- 🌾 Sugar cane, Maize (corn), Rice, and Wheat dominate the dataset, contributing to over 70% of the total production records.
- 🌍 China (including China mainland), Mexico and Peru have the highest number of records in the dataset.
- 🍀 The crop diversity includes both staple food grains and high-yield commercial crops, indicating a globally diverse production dataset.
    """)

# Trend Analysis
elif page == "Trend Analysis":
    st.title("📈 Trend Analysis")

    df['Year'] = df['Year'].astype(int)

    crop = st.selectbox("Select Crop", df['Item'].unique())
    area = st.selectbox("Select Area", df['Area'].unique())
    df_filtered = df[(df['Item'] == crop) & (df['Area'] == area)]  

    if df_filtered.empty:
        st.warning("No data for this selection")
    else:
        fig, ax = plt.subplots()
        sns.lineplot(data=df_filtered, x='Year', y='Production_tons', marker='o', ax=ax)

        # 🔧 Force integer ticks on the x-axis
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set_title(f"Production over Time - {crop} in {area}")
        st.pyplot(fig)

# Predict Production
elif page == "Predict Production":
    st.title("🤖 Predict Crop Production")
    area = st.selectbox("Area", sorted(df['Area'].unique()))
    item = st.selectbox("Crop", sorted(df['Item'].unique()))
    year = st.number_input("Year", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), step=1)
    area_harvested = st.number_input("Area Harvested (ha)", min_value=1.0)
    yield_kg = st.number_input("Yield (kg/ha)", min_value=1.0)

    if st.button("Predict"):
        # Encode manually (dummy-style)
        input_df = pd.DataFrame({
            'Year': [year],
            'Area_Harvested': [area_harvested],
            'Yield_kg_per_ha': [yield_kg],
            'Area_' + area: [1],
            'Item_' + item: [1]
        })

        # Add missing columns as 0
        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model.feature_names_in_]
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Production: {prediction:,.2f} tons")

# Crop Distribution
elif page == "Crop Distribution":
    st.title("🌿 Crop Distribution")

    # -------------------- Most Cultivated Crops --------------------
    st.subheader("Most Cultivated Crops")
    top_crops = df.groupby('Item')['Area_Harvested'].sum().sort_values(ascending=False).head(10)
    top_crops_million = top_crops / 1_000_000
    top_crops_million_sorted = top_crops_million.sort_values()

    fig, ax = plt.subplots(figsize=(24, 18))
    top_crops_million_sorted.plot(kind='barh', ax=ax, color='green', width=0.7)

    ax.set_xlabel("Area Harvested (in millions of hectares)", fontsize=26, color='black')
    ax.set_ylabel("Crop (Item)", fontsize=26, color='black')
    ax.set_title("Top 10 Most Cultivated Crops by Area Harvested", fontsize=30, color='black')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.tick_params(axis='both', labelsize=24, colors='black')

    for i, value in enumerate(top_crops_million_sorted.values):
        ax.text(value + 0.1, i, f'{value:.1f} M ha', va='center', fontsize=22, color='black')

    st.pyplot(fig)

    # -------------------- Least Cultivated Crops --------------------
    st.subheader("Least Cultivated Crops")

    # Bottom 10 crops by Area Harvested
    least_crops = df.groupby('Item')['Area_Harvested'].sum().nsmallest(10)
    least_crops_million = least_crops / 1_000_000

    fig, ax = plt.subplots(figsize=(24, 18))
    least_crops_million.plot(kind='barh', ax=ax, color='red', width=0.7)

    ax.set_xlabel("Area Harvested (in millions of hectares)", fontsize=26, color='black')
    ax.set_ylabel("Crop (Item)", fontsize=26, color='black')
    ax.set_title("Least 10 Cultivated Crops by Area Harvested", fontsize=30, color='black')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))

    # Tick styling
    ax.tick_params(axis='both', labelsize=24, colors='black')

    # Value labels (aligned close to bars)
    for i, value in enumerate(least_crops_million.values):
        ax.text(value + max(least_crops_million.values) * 0.02, i, f'{value:.2f} M ha',
            va='center', fontsize=22, color='black')

    st.pyplot(fig)

    # -------------------- Geographical Crop Focus --------------------
    st.subheader("Geographical Crop Focus")
    top_areas = df.groupby('Area')['Production_tons'].sum().sort_values(ascending=False).head(10)
    top_areas_million = top_areas / 1_000_000
    top_areas_million_sorted = top_areas_million.sort_values()

    fig, ax = plt.subplots(figsize=(24, 18))
    top_areas_million_sorted.plot(kind='barh', ax=ax, color='orange', width=0.7)

    ax.set_xlabel("Production in million tonnes", fontsize=26, color='black')
    ax.set_ylabel("Area (Region)", fontsize=26, color='black')
    ax.set_title("Top 10 Regions by Crop Production", fontsize=30, color='black')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.tick_params(axis='both', labelsize=24, colors='black')

    for i, value in enumerate(top_areas_million_sorted.values):
        ax.text(value + 0.1, i, f'{value:.2f} M tons', va='center', fontsize=22, color='black')

    st.pyplot(fig)

# ------------------ Region-wise Top Crop -------------------
    st.subheader("🔍 Most Produced Crop in Selected Region")

    # Let user pick from the top 10 regions shown earlier
    selected_region = st.selectbox("Select a Region", top_areas_million_sorted.index[::-1])  # Reverse for readability

    # Filter data for the selected region
    region_df = df[df['Area'] == selected_region]

    # Find the crop with highest total production in that region
    top_crop_region = region_df.groupby('Item')['Production_tons'].sum().sort_values(ascending=False).head(1)

    # Display the result
    if not top_crop_region.empty:
        crop_name = top_crop_region.index[0]
        crop_production = top_crop_region.iloc[0] / 1_000_000  # Convert to million tons

        st.success(f"🌾 In **{selected_region}**, the crop with highest production is **{crop_name}** "
               f"with **{crop_production:.2f} million tons**.")
    
        # Optional: visualize top 5 crops in selected region
        top_crops_region = region_df.groupby('Item')['Production_tons'].sum().sort_values(ascending=False).head(5)
        top_crops_region_million = top_crops_region / 1_000_000

        fig, ax = plt.subplots(figsize=(20, 12))
        top_crops_region_million.plot(kind='barh', ax=ax, color='purple')
        ax.set_xlabel("Production (in million tons)", fontsize=20, color='black')
        ax.set_ylabel("Crop", fontsize=20, color='black')
        ax.set_title(f"Top 5 Crops in {selected_region}", fontsize=24, color='black')
        ax.tick_params(axis='both', labelsize=18, colors='black')

        # Add value labels
        for i, value in enumerate(top_crops_region_million.values):
            ax.text(value + 0.1, i, f'{value:.2f}', va='center', fontsize=18, color='black')

        st.pyplot(fig)

    else:
        st.warning("No crop data available for this region.")


# Temporal Analysis
elif page == "Temporal Analysis":
    st.title("🕒 Temporal Analysis")

    # Ensure Year is integer
    df['Year'] = df['Year'].astype(int)
    # Yearly Trends: Area Harvested over Time
    st.subheader("Area Harvested (in hectares) Over Time")

    # Group by Year and calculate total Area Harvested
    area_trend = df.groupby('Year')['Area_Harvested'].sum().reset_index()

    # Convert Year to integer to avoid decimals
    area_trend['Year'] = area_trend['Year'].astype(int)

    # Convert Area Harvested to millions for better readability
    area_trend['Area_Harvested_million'] = area_trend['Area_Harvested'] / 1_000_000

    # Plot using matplotlib for more control
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=area_trend, x='Year', y='Area_Harvested_million', marker='o', color='green', ax=ax)
    ax.set_xlabel("Year", fontsize=16, color='black')
    ax.set_ylabel("Area Harvested (in million hectares)", fontsize=16, color='black')
    ax.set_title("Total Area Harvested Over Time", fontsize=18, color='black')
    #ax.tick_params(axis='both', labelsize=14, colors='black')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xticks(area_trend['Year'])  # Ensures only integer years
    st.pyplot(fig)

    # Average Yield over Time
    st.subheader("Average Yield (in kg per ha) over Time")
    yield_trend = df.groupby('Year')['Yield_kg_per_ha'].mean().reset_index()

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(yield_trend['Year'], yield_trend['Yield_kg_per_ha'], marker='o', color='green')
    ax1.set_title("Average Yield over Time", fontsize=16)
    ax1.set_xlabel("Year", fontsize=14)
    ax1.set_ylabel("Yield (kg per ha)", fontsize=14)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.set_xticks(yield_trend['Year'])  # Ensures only integer years
    st.pyplot(fig1)

    # Crop Growth Analysis
    st.subheader("Crop Growth Analysis with Production (in tons) over Time")
    crop_growth = df.groupby(['Year', 'Item'])['Production_tons'].sum().reset_index()

    crop = st.selectbox("Choose Crop", df['Item'].unique())
    subset = crop_growth[crop_growth['Item'] == crop]

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(subset['Year'], subset['Production_tons'], marker='o', color='blue')
    ax2.set_title(f"Production of {crop} over Time", fontsize=16)
    ax2.set_xlabel("Year", fontsize=14)
    ax2.set_ylabel("Production (tons)", fontsize=14)
    ax2.set_xticks(subset['Year'])  # Force X-axis to show only integer years
    ax2.tick_params(axis='x', labelrotation=45)
    st.pyplot(fig2)

    # Growth Analysis Section
    st.subheader("Growth Analysis: Crop-wise or Region-wise Trends")

    # Give option to select analysis type
    analysis_type = st.radio("Select Analysis Type", ("Crop-wise", "Region-wise"))

    if analysis_type == "Crop-wise":
        crop = st.selectbox("Select Crop", df['Item'].unique())
    
        # Filter data
        crop_df = df[df['Item'] == crop].groupby('Year').agg({
            'Production_tons': 'sum',
            'Yield_kg_per_ha': 'mean'
        }).reset_index()
    
        crop_df['Year'] = crop_df['Year'].astype(int)  # Ensure Year is integer

        # Plot Production trend
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=crop_df, x=crop_df['Year'], y='Production_tons', marker='o', color='orange', ax=ax)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Production (tons)", fontsize=14)
        ax.set_title(f"Production Trend for {crop}", fontsize=16)
        ax.set_xticks(subset['Year'])  # Force X-axis to show only integer years
        ax.tick_params(axis='x', labelrotation=45)
        #ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)

        # Plot Yield trend
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=crop_df, x=crop_df['Year'], y='Yield_kg_per_ha', marker='o', color='green', ax=ax)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Yield (kg/ha)", fontsize=14)
        ax.set_title(f"Yield Trend for {crop}", fontsize=16)
        ax.set_xticks(subset['Year'])  # Force X-axis to show only integer years
        ax.tick_params(axis='x', labelrotation=45)
        #ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)

    elif analysis_type == "Region-wise":
        region = st.selectbox("Select Region", df['Area'].unique())
    
        region_df = df[df['Area'] == region].groupby('Year').agg({
            'Production_tons': 'sum',
            'Yield_kg_per_ha': 'mean'
        }).reset_index()

        region_df['Year'] = region_df['Year'].astype(int)  # Ensure Year is integer

        # Plot Production trend
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=region_df, x=region_df['Year'], y='Production_tons', marker='o', color='orange', ax=ax)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Production (tons)", fontsize=14)
        ax.set_title(f"Production Trend for {region}", fontsize=16)
        ax.set_xticks(subset['Year'])  # Force X-axis to show only integer years
        ax.tick_params(axis='x', labelrotation=45)
        #ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)

        # Plot Yield trend
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=region_df, x=region_df['Year'], y='Yield_kg_per_ha', marker='o', color='green', ax=ax)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Yield (kg/ha)", fontsize=14)
        ax.set_title(f"Yield Trend for {region}", fontsize=16)
        ax.set_xticks(subset['Year'])  # Force X-axis to show only integer years
        ax.tick_params(axis='x', labelrotation=45)
        #ax.tick_params(axis='both', labelsize=12)
        st.pyplot(fig)


# Environmental Relationships 

# Environmental Relationships
elif page == "Environmental Relationships":
    st.title("🌱 Environmental Inference")
    st.subheader("Scatterplot: Area Harvested vs Yield")

    # Create transformed columns
    df['Area_Harvested_Million'] = df['Area_Harvested'] / 1_000_000
    df['Yield_kg_per_million_ha'] = df['Yield_kg_per_ha'] / 1_000_000

    fig, ax = plt.subplots(figsize=(24, 16))  # Much bigger plot

    sns.scatterplot(
        data=df, 
        x='Area_Harvested_Million', 
        y='Yield_kg_per_million_ha', 
        hue='Item', 
        ax=ax, 
        s=200,  # increase marker size
        edgecolor='black', 
        legend=False
    )

    ax.set_xlabel("Area Harvested (in million hectares)", fontsize=28, color='black')
    ax.set_ylabel("Yield (kg per million hectares)", fontsize=28, color='black')
    ax.set_title("Area Harvested vs Yield (Scaled View)", fontsize=32, color='black')
    ax.tick_params(axis='both', labelsize=24, colors='black')

    # Format axes to avoid exponential notation and show decimals
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2f}'))

    st.pyplot(fig)
    # Insights Box for Environmental Inference
    with st.expander("📌 Insights from Environmental Scatterplot", expanded=True):
        st.markdown("""
        **Key Observations:**

        - 🌿 **Inverse relationship between Area Harvested and Yield:**  
        ➔ As Area Harvested increases, Yield per hectare tends to be lower. This indicates that large-scale cultivation may be compromising yield efficiency.
    
        - 🔬 **Significant clustering at low area and low yield:**  
        ➔ A majority of data points are concentrated where both area harvested and yield are relatively low, suggesting potential inefficiencies or under-utilized production capacity.

        - 🚩 **High-yield outliers at smaller harvested areas:**  
        ➔ Few regions or crops are achieving exceptionally high yields in small cultivated areas, indicating best practices or favorable agro-climatic conditions.

        **Actionable Recommendations:**

        - 🌱 **Adopt Precision Farming:**  
        ➔ Utilize modern techniques like soil testing, crop rotation, optimized irrigation, and targeted fertilization to maximize yield even in smaller plots.

        - 📊 **Focus on Best Practices Replication:**  
        ➔ Identify high-yield regions and analyze their practices to replicate success across other areas.

        - 🌎 **Sustainable Expansion:**  
        ➔ Rather than merely expanding area, focus on optimizing productivity per hectare to achieve long-term sustainability.

        """, unsafe_allow_html=True)


# Input-Output Relationships
elif page == "Input-Output Relationships":
    st.title("🔄 Input vs Output")
    st.subheader("Correlation Heatmap")

    # Calculate correlation
    corr = df[['Area_Harvested', 'Yield_kg_per_ha', 'Production_tons']].corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', square=True, cbar_kws={'shrink': 0.7}, ax=ax)
    st.pyplot(fig)

    # Insights Box
    with st.expander("📌 Insights from Correlation Heatmap", expanded=True):
        st.markdown("""
        **Key Observations:**

        - 🔗 **Strong correlation (0.64) between Area Harvested and Production**  
          ➔ Expanding cultivated area increases production, but expansion may be limited due to environmental or land availability constraints.
        
        - 📉 **Very weak correlation (0.05) between Yield and Production**  
          ➔ Yield improvements have not significantly contributed to overall production so far. There is considerable untapped potential to increase production through yield enhancement.
        
        - ⚖️ **No significant correlation (-0.03) between Area Harvested and Yield**  
          ➔ Yield improvements depend more on farming practices, technology, and inputs than on the size of land cultivated.

        **Actionable Recommendations:**

        - 🌾 **Prioritize Yield Improvement Programs**  
          ➔ Invest in better seeds, fertilizers, irrigation, and modern farming techniques to boost yield efficiency.
        
        - 🗺️ **Focus on High-Yield Regions**  
          ➔ Identify and replicate best practices from high-yielding regions across other areas.
        
        - 🌿 **Optimize Existing Land Use**  
          ➔ Focus on improving productivity of currently cultivated areas rather than expanding into new lands.
        
        - 🔬 **Conduct Detailed Studies**  
          ➔ Further research can identify specific regional or crop-level bottlenecks to unlock yield potential.
        """, unsafe_allow_html=True)

# Comparative Analysis
elif page == "Comparative Analysis":
    st.title("📊 Comparative Analysis")
    st.subheader("Average Yield per Crop")
    crop_yield = df.groupby('Item')['Yield_kg_per_ha'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(crop_yield)

    st.subheader("Top Producing Regions")
    top_regions = df.groupby('Area')['Production_tons'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_regions)

# Outliers and Anomalies
elif page == "Outliers and Anomalies":
    st.title("🚨 Outliers & Anomalies")

    # Yield Boxplot
    st.subheader("Yield Boxplot")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='Yield_kg_per_ha', ax=ax, color='skyblue')
    ax.set_xlabel("Yield (kg per ha)", fontsize=14, color='black')
    ax.tick_params(axis='x', colors='black', labelsize=12)
    st.pyplot(fig)

    # Production Boxplot (in millions of tons)
    st.subheader("Production Boxplot")

    # Convert Production to millions of tons
    df['Production_million_tons'] = df['Production_tons'] / 1_000_000

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='Production_million_tons', ax=ax, color='lightgreen')

    ax.set_xlabel("Production (Million Tons)", fontsize=14, color='black')
    ax.tick_params(axis='x', colors='black', labelsize=12)

    # Format x-axis to show fixed point with 1 decimal place
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))

    st.pyplot(fig)
    st.markdown("""
    ### 📊 **Insights from Yield and Production Boxplots**

    - Most crop yields are concentrated in a narrow range, indicating consistency for majority of crops.
    - A few extreme high-yield values suggest certain regions or techniques achieve exceptional yields. These outliers can be studied for best practices.
    - Very high production outliers indicate heavy concentration in a few major crops like wheat, rice, maize, or sugarcane.
    - The presence of many low-production crops highlights the need for diversification to strengthen food security.
    - Overall, the variability suggests strong potential for improvement in both yield and production with targeted interventions.
    - Extreme values should also be reviewed for possible data anomalies or entry errors.
    """)


# Actionable Insights
elif page == "Actionable Insights":
    st.title("📊 Actionable Insights")
    st.subheader("Data-Driven Recommendations for Agricultural Planning")

    st.markdown("### 1️⃣ Focus on High Yield Crops")
    high_yield_crops = df.groupby('Item')['Yield_kg_per_ha'].mean().sort_values(ascending=False).head(5)
    st.write("Crops with highest average yield:")
    st.dataframe(high_yield_crops)

    st.markdown("### 2️⃣ Regions with High Production")
    high_production_regions = df.groupby('Area')['Production_tons'].sum().sort_values(ascending=False).head(5)
    st.write("Regions with highest total production:")
    st.dataframe(high_production_regions)

    st.markdown("### 3️⃣ Under-utilized Crops (Low Area Harvested but High Yield)")
    # Find crops with low area but high yield (opportunity)
    avg_area = df.groupby('Item')['Area_Harvested'].mean()
    avg_yield = df.groupby('Item')['Yield_kg_per_ha'].mean()
    opportunity_df = pd.DataFrame({'Area_Harvested': avg_area, 'Yield_kg_per_ha': avg_yield})
    opportunity_df = opportunity_df[opportunity_df['Area_Harvested'] < opportunity_df['Area_Harvested'].mean()]
    opportunity_df = opportunity_df.sort_values(by='Yield_kg_per_ha', ascending=False).head(5)
    st.write("Crops with low harvested area but high yield potential:")
    st.dataframe(opportunity_df)

    st.markdown("### 🔎 Recommendations:")
    st.write("""
    - Allocate more resources (land, irrigation, research) to crops with high yield potential but low area harvested.
    - Focus infrastructure and supply chains in regions with highest production.
    - Monitor crops that are showing consistent growth trends for long-term sustainability.
    - Encourage diversification by supporting least cultivated but high yield crops.
    """)
