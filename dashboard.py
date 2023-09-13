
import streamlit as st
import pandas as pd
import pydeck as pdk  # Add this import
import altair as alt
import geopandas as gpd
from streamlit_folium import folium_static
import folium
from shapely.geometry import Point
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabulate import tabulate
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import matplotlib.dates as mdates
from wordcloud import WordCloud




# Get the data
url = "https://drive.google.com/file/d/1V37XgiU9OBT35M83LgFoLTxDKfp8BTej/view?usp=drive_link"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
data = pd.read_csv(path)




# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set the page title and other configurations
st.set_page_config(page_title='Fast-Food Brands Dashboard')

# Sidebar filters
st.sidebar.title('Filter Options')

# Filter by market, brand, and date
selected_date_range = st.sidebar.date_input('Select Date Range', [data['Date'].min(), data['Date'].max()], key="select_date_range")
selected_markets = st.sidebar.multiselect('Select Market', data['Market1'].unique(), key="select_markets")
selected_brands = st.sidebar.multiselect('Select Brand', data['Brand name'].unique(), key="select_brands")

# Convert the selected date range to Timestamp
selected_start_date = pd.to_datetime(selected_date_range[0])
selected_end_date = pd.to_datetime(selected_date_range[1])

# Filter data for selected markets, brands, and date range
filtered_data = data[(data['Market1'].isin(selected_markets)) & 
                     (data['Brand name'].isin(selected_brands)) & 
                     (data['Date'] >= selected_start_date) & 
                     (data['Date'] <= selected_end_date)]

# Calculate Total Budget and Total Volume
total_budget = filtered_data['Budget'].sum()
total_volume = filtered_data['Volume'].sum()

# Summary in sidebar
st.sidebar.subheader('Summary')
st.sidebar.write(f"Selected Markets: {', '.join(selected_markets)}")
st.sidebar.write(f"Selected Brands: {', '.join(selected_brands)}")
st.sidebar.write(f"Total Budget: {total_budget}")
st.sidebar.write(f"Total Volume: {total_volume}")

# Check if the filtered data is empty
if filtered_data.empty:
    st.warning("No data available for the selected filters.")
else:
    # Header
    st.title('Fast-Food Brands Dashboard')

    # Line chart of Total Budget over time
    total_budget_over_time = data.groupby('Date')['Budget'].sum().reset_index()

    # Radio button to select period
    st.subheader("Line chart of Total Budget over time")
    period_option = st.radio("Select Period:", ['daily', 'weekly', 'monthly', 'quarterly', 'yearly'])

    # Resample the data based on the selected period
    if period_option == 'daily':
        resample_rule = 'D'
        x_axis_format = '%Y-%m-%d'
    elif period_option == 'weekly':
        resample_rule = 'W-MON'
        x_axis_format = '%Y-%m-%d'
    elif period_option == 'monthly':
        resample_rule = 'M'
        x_axis_format = '%Y-%m'
    elif period_option == 'quarterly':
        resample_rule = 'Q'
        x_axis_format = '%Y-Q%q'
    elif period_option == 'yearly':
        resample_rule = 'Y'
        x_axis_format = '%Y'

    total_budget_resampled = total_budget_over_time.resample(resample_rule, on='Date').sum().reset_index()

    # Line chart with resampled data
    total_budget_chart = alt.Chart(total_budget_resampled).mark_line().encode(
        x=alt.X('Date:T', axis=alt.Axis(format=x_axis_format, title='Date')),
        y=alt.Y('Budget:Q', axis=alt.Axis(title='Total Budget')),
        tooltip=['Date', 'Budget']
    ).properties(
        width=800,
        height=400,
        title='Total Budget Over Time'
    ).interactive()

    # Display the Radio button and Line chart side by side
    col1, col2 = st.columns([2, 1])
    with col1:
        st.altair_chart(total_budget_chart)

    with col2:
        st.sidebar.subheader("Select Period:")
        period_option = st.sidebar.radio("Choose a period:", ['daily', 'weekly', 'monthly', 'quarterly', 'yearly'])
    
    # ...

    # Brand Budget Comparison chart
    st.subheader('Brand Budget Comparison')
    brand_budget_comparison = filtered_data.groupby(['Date', 'Brand name'])['Budget'].sum().reset_index()
    brand_budget_chart = alt.Chart(brand_budget_comparison).mark_bar().encode(
        x='Date:T',
        y='Budget:Q',
        color='Brand name:N',
        tooltip=['Date', 'Brand name', 'Budget']
    ).properties(
        width=800,
        height=400,
        title='Brand Budget Comparison'
    ).interactive()
    st.altair_chart(brand_budget_chart)

    # EDA charts for Market1 by brand
    st.subheader('Market EDA by Brand')
    brand_color_scale = alt.Scale(domain=data['Brand name'].unique(),
                                  range=['red', 'green', 'blue', 'orange', 'purple'])

    market_data = filtered_data[['Date', 'Market1', 'Brand name', 'Budget']]
    market_data['Date'] = pd.to_datetime(market_data['Date'])
    line_chart = alt.Chart(market_data).mark_line().encode(
        x='Date:T',
        y=alt.Y('Budget:Q', title='Budget'),
        color=alt.Color('Brand name:N', scale=brand_color_scale),
        column='Market1:N'
    ).properties(
        width=300,
        height=200,
        title='Budget Over Time by Brand and Market'
    )
    st.altair_chart(line_chart)

    # ...
    # Rest of your code for other visualizations




    # Top visuals by size based on sum of budget
top_visuals_count = st.slider("Select the number of top visuals:", 1, 10, 5)
top_visuals_by_size = filtered_data.groupby('Visual')['Budget'].sum().nlargest(top_visuals_count).reset_index()

# Create a horizontal bar chart
top_visuals_chart = alt.Chart(top_visuals_by_size).mark_bar().encode(
    y=alt.Y('Visual:N', sort='-x'),  # Sort the visuals in descending order
    x='Budget:Q',
    color='Visual:N',
    tooltip=['Visual', 'Budget']
).properties(
    width=800,
    height=400,
    title=f'Top {top_visuals_count} Visuals by Budget'
).interactive()

# Display the chart
st.subheader('Top Visuals by Budget')
st.altair_chart(top_visuals_chart)


# Calculate the percentage of campaigns by device
campaigns_by_device_percent = (filtered_data['Platform'].value_counts() / filtered_data['Platform'].count()) * 100

# Interactive Number Text Visual for Campaigns by Device
st.subheader('Campaigns by Device')
selected_platform = st.selectbox('Select a Device', campaigns_by_device_percent.index)
if selected_platform in campaigns_by_device_percent.index:
    selected_percentage = campaigns_by_device_percent[selected_platform]
    st.write(f"<p style='font-size:24px;'>Percentage of Campaigns on {selected_platform}: <span style='color:blue;'>{selected_percentage:.2f}%</span></p>", unsafe_allow_html=True)
else:
    st.write("Please select a valid device.")






# Calculate Total Budget and Total Volume
total_budget = filtered_data['Budget'].sum()
total_volume = filtered_data['Volume'].sum()

# Language Identification
language_counts = filtered_data['Language'].value_counts()

# Merge 'Music' and '-' with 'OTHER' category
if 'Music' in language_counts:
    language_counts['OTHER'] = language_counts.get('OTHER', 0) + language_counts.pop('Music')
if '-' in language_counts:
    language_counts['OTHER'] = language_counts.get('OTHER', 0) + language_counts.pop('-')

language_identification = pd.DataFrame({'Language': language_counts.index, 'Count': language_counts.values})

# Update color scale to include "OTHER"
language_color_scale = alt.Scale(
    domain=language_identification['Language'].unique(),
    range=['red', 'green', 'blue', 'orange', 'purple', 'gray']
)

# Create the language identification pie chart
language_pie_chart = alt.Chart(language_identification).mark_bar().encode(
    x=alt.X('Count:Q', axis=alt.Axis(title='Count')),
    y=alt.Y('Language:N', sort='-x'),
    color=alt.Color('Language:N', scale=language_color_scale),
    tooltip=['Language', 'Count']
).properties(
    width=800,
    height=500,
    title='Language Identification'
).interactive()

# Display the language identification pie chart
st.altair_chart(language_pie_chart)




# Interactive Bar Chart for Campaign Status
campaign_status_counts = filtered_data['Campaign status'].value_counts()
campaign_status_chart = alt.Chart(pd.DataFrame({'Campaign Status': campaign_status_counts.index, 'Count': campaign_status_counts.values})).mark_bar().encode(
    x='Campaign Status:N',
    y='Count:Q',
    color='Campaign Status:N',
    tooltip=['Campaign Status', 'Count']
).properties(
    width=800,
    height=400,
    title='Campaign Status Distribution'
).interactive()
st.altair_chart(campaign_status_chart)

# WORD CLOUD CODE
# Generate a word cloud for campaign names if there is data available
if not filtered_data['Campaign'].empty:
    # Create a dictionary to map campaign names to their corresponding budgets
    campaign_budgets = dict(zip(filtered_data['Campaign'], filtered_data['Budget']))

    # Generate the WordCloud with size based on campaign budgets
    wordcloud = WordCloud(width=800, height=400, background_color='white', prefer_horizontal=1, relative_scaling=0.5, max_words=200)
    wordcloud.generate_from_frequencies(campaign_budgets)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Visualizing Campaign Themes through Budget-Weighted Word Cloud", fontsize=16, fontweight='bold')
    st.pyplot(plt)
else:
    st.write("No campaign names available for generating a word cloud.")




# Interactive Scatter Plot for Budget vs Volume
budget_volume_scatter = alt.Chart(filtered_data).mark_circle(size=60).encode(
    x='Budget:Q',
    y='Volume:Q',
    color='Brand name:N',
    tooltip=['Brand name', 'Budget', 'Volume']
).properties(
    width=800,
    height=400,
    title='Budget vs Volume'
).interactive()
st.altair_chart(budget_volume_scatter)




# Interactive Heatmap
heatmap_data = filtered_data.groupby(['Brand name', 'Month', 'Campaign status']).size().reset_index(name='Count')

heatmap_chart = alt.Chart(heatmap_data).mark_rect().encode(
    x='Month:N',
    y='Brand name:N',
    color='Count:Q',
    tooltip=['Brand name', 'Month', 'Campaign status', 'Count']
).properties(
    width=800,
    height=400,
    title='Campaign Status by Brand and Month'
).interactive()

st.altair_chart(heatmap_chart)








# ...

# Conclusions and Final Remarks Section
st.subheader('Conclusions and Final Remarks')

# Conclusions about Market1 EDA
st.write("The Market1 EDA by Brand visualizations offer valuable insights into the competitive landscape "
         "of fast food brands in the region. The budget distribution over time indicates each brand's "
         "investment patterns, which can inform strategic decisions to align with market trends.")

# Conclusions about Top Visuals by Size
st.write("The Top 10 Visuals by Size chart provides insights into the effectiveness of different visual sizes "
         "in engaging the audience. This knowledge is crucial for creating impactful marketing materials that "
         "capture the attention of consumers across various platforms.")

# Conclusions about Campaigns by Device
st.write("The Campaigns by Device chart highlights the platforms that brands are leveraging for their campaigns. "
         "Understanding these preferences can guide media planning decisions, ensuring campaigns are optimized "
         "for the devices most used by the target audience.")

# Conclusions about Language Identification
st.write("Language Identification insights shed light on the languages preferred by the audience. Given the diverse "
         "nature of the region, catering to multiple languages such as English, Spanish, and French can help brands "
         "reach a broader spectrum of consumers.")

# Overall Insights
st.write("In summary, this project equips fast food brands with data-backed insights to navigate the competitive "
         "media landscape effectively. By analyzing budget allocation, visual effectiveness, platform preferences, "
         "and language usage, brands can refine their marketing strategies for better brand visibility and audience "
         "engagement.")

# Recommendations
st.write("Based on these findings, we recommend brands to continuously monitor budget allocation trends to stay "
         "competitive. Prioritizing larger visuals and optimizing campaigns for preferred devices and languages "
         "can lead to higher engagement rates and brand resonance.")

# Final Remarks
st.write("In conclusion, this dashboard provides actionable insights for McDonald's, KFC, Pizza Hut, Burger King, "
         "and Hardee's to strategically position themselves in the fast food market. By harnessing data-driven "
         "decisions, these brands can capture consumer attention and foster brand loyalty in a dynamic media landscape.")




# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data for the brands you are interested in
selected_brands = ["MCDONALD'S", 'KFC', 'PIZZA HUT', 'BURGER KING', "HARDEE'S"]
filtered_data = data[data['Brand name'].isin(selected_brands)]

# Group data by month and brand, and calculate the total budget
monthly_brand_budget = filtered_data.groupby(['Brand name', pd.Grouper(key='Date', freq='M')])['Budget'].sum().reset_index()

# Set the forecast steps for Q1 2023
forecast_steps = 3

# Create the Streamlit app
st.title("Machine Learning Dashboard")

# Sidebar for user interaction
st.sidebar.header("Task Selection")
selected_task = st.sidebar.radio("Select Task", ["Forecasting", "Metrics Comparison", "Process Flow", "Data Analysis","Data Cleaning"])

if selected_task == "Forecasting":
    st.subheader("Budget Forecasting")

    # Define a function to train and forecast with SARIMA model
    def sarima_forecast(series, order, seasonal_order, forecast_steps):
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=forecast_steps)
        return forecast

    # Display brand selection for forecasting
    st.sidebar.subheader("Select Brands for Forecasting")
    selected_brands_forecast = st.sidebar.multiselect("Select Brands", selected_brands)

    if selected_brands_forecast:
        # Iterate through selected brands and train SARIMA models
        for brand in selected_brands_forecast:
            st.write(f"### Forecasting for {brand}")

            brand_data = monthly_brand_budget[monthly_brand_budget['Brand name'] == brand]

            # Split the data into training and testing sets
            train_data = brand_data.iloc[:-forecast_steps]
            test_data = brand_data.iloc[-forecast_steps:]

            # Define SARIMA parameters (you may need to fine-tune these)
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)  # 12 for monthly data

            # Train and forecast with SARIMA model
            forecast = sarima_forecast(train_data['Budget'], order, seasonal_order, forecast_steps)

            # Display forecast results
            st.write("Forecasted Budget for Q1 2023:")
            st.write(forecast)

            # Plot historical and forecasted budget data
            plt.figure(figsize=(10, 6))
            plt.plot(train_data['Date'], train_data['Budget'], label='Historical Budget', marker='o')
            plt.axvline(x=test_data['Date'].iloc[0], color='r', linestyle='--', label='Forecast Start')
            plt.plot(test_data['Date'], forecast, label='Forecasted Budget', color='g', marker='x')
            plt.title(f'Budget Forecast for {brand}')
            plt.xlabel('Date')
            plt.ylabel('Budget')
            plt.legend()
            st.pyplot(plt)



            # Create a scatter plot to visualize the actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(test_data['Date'], test_data['Budget'], color='blue', label='Actual', marker='o')
            plt.scatter(test_data['Date'], forecast, color='red', label='Predicted', marker='x')
            plt.title('Actual vs. Predicted Budget')
            plt.xlabel('Date')
            plt.ylabel('Budget')
            plt.legend()

            # Format the date labels
            date_format = mdates.DateFormatter('%Y-%m-%d')  # Customize the date format here
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.gcf().autofmt_xdate()  # Automatically adjust the x-axis date labels

            plt.grid(True)
            st.pyplot(plt)     
            

# Define a function to train and forecast with ARIMA model
def arima_forecast(series, order, seasonal_order, forecast_steps):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    return forecast


if selected_task == "Forecasting":
    st.subheader("Budget Forecasting")

    # Define forecast months for Q1 2023
    forecast_months = ["January", "February", "March"]

    # ...

    if selected_brands_forecast:
        # ...

        # Train and forecast with ARIMA model
        arima_forecast = arima_forecast(train_data['Budget'], order, seasonal_order, forecast_steps)

        # Display forecasted budget for Q1 2023:
        st.write("### Total Forecasted Budget for all Brands - Q1 2023:")
        for i, forecast in enumerate(arima_forecast):
            st.write(f"{forecast_months[i]}: {forecast:.2f}")

        # ...

        plt.grid(True)
        st.pyplot(plt)



elif selected_task == "Metrics Comparison":
    st.subheader("Metrics Comparison")

    # Define function to calculate metrics
    def calculate_metrics(actual, predicted):
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = mse ** 0.5
        return mse, mae, rmse

    # Display brand selection for metrics comparison
    st.sidebar.subheader("Select Brands for Metrics Comparison")
    selected_brands_metrics = st.sidebar.multiselect("Select Brands", selected_brands)

    if selected_brands_metrics:
        # Initialize lists to store results
        results_arima = []
        results_sarima = []

        # Iterate through selected brands
        for brand in selected_brands_metrics:
            brand_data = monthly_brand_budget[monthly_brand_budget['Brand name'] == brand]

            # Split the data into training and testing sets
            train_data = brand_data.iloc[:-forecast_steps]
            test_data = brand_data.iloc[-forecast_steps:]

            # Fit ARIMA model
            arima_model = ARIMA(train_data['Budget'], order=(5, 1, 0))
            arima_fit = arima_model.fit()

            # Fit SARIMA model
            sarima_model = SARIMAX(train_data['Budget'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_fit = sarima_model.fit()

            # Forecast with ARIMA and SARIMA models
            arima_forecast = arima_fit.forecast(steps=len(test_data))
            sarima_forecast = sarima_fit.forecast(steps=len(test_data))

            # Calculate metrics for ARIMA and SARIMA models
            mse_arima, mae_arima, rmse_arima = calculate_metrics(test_data['Budget'], arima_forecast)
            mse_sarima, mae_sarima, rmse_sarima = calculate_metrics(test_data['Budget'], sarima_forecast)

            # Store results in lists
            results_arima.append([brand, mse_arima, mae_arima, rmse_arima])
            results_sarima.append([brand, mse_sarima, mae_sarima, rmse_sarima])

        # Create dataframes for results
        columns = ['Brand', 'MSE', 'MAE', 'RMSE']
        df_results_arima = pd.DataFrame(results_arima, columns=columns)
        df_results_sarima = pd.DataFrame(results_sarima, columns=columns)

        # Display results in table format
        st.write("### ARIMA Model Metrics:")
        st.write(df_results_arima)

        st.write("### SARIMA Model Metrics:")
        st.write(df_results_sarima)

        # Display forecasted budget for Q1 2023
        st.write("### Forecasted Budget for Q1 2023:")
        forecast_months = ["January", "February", "March"]
        for i, forecast in enumerate(arima_forecast):
            st.write(f"{forecast_months[i]}: {forecast:.2f}")
            


elif selected_task == "Process Flow":
    st.subheader("Process Flow")

    # Load and display an image of the process flow
    img = Image.open('Process_flow.png')

    st.image(img, caption='Process Flow', use_column_width=True)

   # Add a section for Data Analysis
elif selected_task == "Data Analysis":
    st.subheader("Data Analysis")


    # Load the raw data before annotation
    url1 = "https://drive.google.com/file/d/1tU4GbXiw86U_YSWNQs2QIsEZI0BRlmAE/view?usp=drive_link"
    path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]
    raw_data = pd.read_csv(path1)


    # Load the data after annotation
    labeled_data = data

    # Display the first 5 rows of the raw data
    st.write("First 5 rows of Raw Data:")
    st.write(raw_data.head())

    # Display the first 5 rows of the labeled data
    st.write("\nFirst 5 rows of labeled Data:")
    st.write(labeled_data.head())

    # Plot the raw data
    st.write("First 5 Rows of Raw Data:")
    plt.figure(figsize=(10, 6))
    raw_data.head().plot(kind='bar')
    plt.title('First 5 Rows of Raw Data')
    plt.xlabel('Index')
    plt.ylabel('Values')
    st.pyplot(plt)

    # Plot the labeled data
    st.write("First 5 Rows of labeled Data:")
    plt.figure(figsize=(10, 6))
    labeled_data.head().plot(kind='bar')
    plt.title('First 5 Rows of labeled Data')
    plt.xlabel('Index')
    plt.ylabel('Values')
    st.pyplot(plt)






# Add a section for Data Cleaning
elif selected_task == "Data Cleaning":
    st.subheader("Data Cleaning")

    # Display unique brand names and their distinct counts in a table
    unique_brands = data["Brand name"].unique()
    distinct_count = data["Brand name"].value_counts()
    
    # Create a DataFrame to display both unique brand names and distinct counts
    brand_info = pd.DataFrame({"Unique Brand Names": unique_brands, "Distinct Count": distinct_count})
    st.write("Brand Information:")
    st.write(brand_info)

    # Plot the distinct counts
    plt.figure(figsize=(10, 6))
    distinct_count.plot(kind='bar')
    plt.xlabel('Brand')
    plt.ylabel('Count')
    plt.title('Distinct Count of Brands')
    plt.xticks(rotation=45)
    st.pyplot(plt)


# Add a section for Data Cleaning
elif selected_task == "Data Cleaning":
    st.subheader("Data Cleaning")

    # Display unique brand names and their distinct counts in a table
    unique_brands = data["Brand name"].unique()
    distinct_count = data["Brand name"].value_counts()
    
    # Create a DataFrame to display both unique brand names and distinct counts
    brand_info = pd.DataFrame({"Unique Brand Names": unique_brands, "Distinct Count": distinct_count})
    st.write("Brand Information:")
    st.write(brand_info)

    # Plot the distinct counts
    plt.figure(figsize=(10, 6))
    distinct_count.plot(kind='bar')
    plt.xlabel('Brand')
    plt.ylabel('Count')
    plt.title('Distinct Count of Brands')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Display the unique brand names and their distinct counts
    st.write("Unique Brand Names:")
    st.write(unique_brands)
    
    st.write("Number of Distinct Values:")
    st.write(distinct_count)


# Run the Streamlit app
if __name__ == "__main__":
    st.write("Welcome to the Machine Learning Dashboard!")


