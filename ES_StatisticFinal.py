
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import altair as alt
import plotly.figure_factory as ff
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
#Data science
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
from streamlit_extras.colored_header import colored_header
#from market_profile import MarketProfile
from datetime import datetime, timedelta
from ta.utils import dropna
from ta.volatility import BollingerBands


# 0.  Futures dataframe to check if the user wants futures or equities


import streamlit as st
import pandas as pd

# Initialize the global variable ticker_symbol  set a default value first
ticker_symbol = "AAPL"
is_a_future=False   # set the first value to false   help decide to display datalong name or not

# Data for Futures Contracts
data_futures = {
    'Symbol': ['^SPX','ES=F', 'MES=F', 'MYM=F', 'YM=F', 'NQ=F', 'MNQ=F', 'RTY=F', 'M2k=F', 'GC=F', 'MGC=F', 'SI=F', 'SIL=F'],
    'Name': ['S&P 500 INDEX','S&P Futures', 'Micro S&P Futures', 'Micro  E-mini Dow Futures', 'Dow Futures', 'Nasdaq Futures',
             'Micro Nasdaq Futures', 'Russell 2000 Futures', 'Micro Russell 2000 Futures', 'Gold',
             'Micro Gold Futures,Apr-2024', 'Silver', 'Micro Silver Futures,Mar-2024'],
    'Class':['Index']+ ['Equities'] * 8 + ['Metals'] * 4,
    'Tick': [0.25,0.25, 0.25, 1, 1, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.005, 0.1],
    'Pertic$': [1,12.5, 1.25, 0.5, 5, 5, 0.5, 5, 0.5, 10, 1, 25, 1]
}
# Creating the DataFrame for Futures Contracts
futures = pd.DataFrame(data_futures)

# Data for Stocks
data_stocks = {
    'Symbol': ['AAPL', 'AMZN', 'META', 'GOOGL', 'PG', 'KO', 'AMD', 'DIS', 'NKE', 'MARA'],
    'Name': ['Apple Inc.', 'Amazon.com Inc.', 'Meta Platforms Inc.', 'Alphabet Inc.', 
             'Procter & Gamble Company', 'The Coca-Cola Company', 'Advanced Micro Devices Inc.', 
             'The Walt Disney Company', 'Nike Inc.', 'Marathon Digital Holdings Inc.']
}
# Creating the DataFrame for Stocks
stocks = pd.DataFrame(data_stocks)

def select_ticker():
    global ticker_symbol
    global is_a_future
    
    # Selecting between Futures Contracts and Stocks
    selected_category = st.sidebar.selectbox('Select a category', ['Futures Contracts', 'Stocks'])

    if selected_category == 'Futures Contracts':
        # Selector for Futures Contracts
        is_a_future=True
        selected_symbol_name = st.sidebar.selectbox('Select a Futures Contract', futures['Name'])
        selected_symbol_value = futures[futures['Name'] == selected_symbol_name]['Symbol'].values[0]
        st.write('You selected:', selected_symbol_name)
        st.write('Symbol Value:', selected_symbol_value)
        ticker_symbol = selected_symbol_value
    else:
        # Selector for Stocks
        selected_option = st.sidebar.selectbox('Select an option', ['Select from list', 'New Ticker Symbol'])

        if selected_option == 'Select from list':
            selected_stock_name = st.sidebar.selectbox('Select a Stock', stocks['Name'])
            selected_stock_value = stocks[stocks['Name'] == selected_stock_name]['Symbol'].values[0]
            st.write('You selected:', selected_stock_name)
            st.write('Symbol Value:', selected_stock_value)
            ticker_symbol = selected_stock_value
        else:
           new_ticker_symbol = st.sidebar.text_input('Enter a new Ticker Symbol', value=ticker_symbol)
           st.write('New Ticker Symbol:', new_ticker_symbol)
           ticker_symbol = new_ticker_symbol # Update ticker_symbol with the new value
            
    return ticker_symbol

# Call the function
ticker_symbol = select_ticker()
st.write('Ticker Symbol selected:', ticker_symbol)


# 1. #####--- GET THE DATA FROM YAHOO FINANCE  ---####

#ticker_symbol="ES=F"
#ticker_symbol="MSFT"

symbol=ticker_symbol
stock = yf.Ticker(ticker_symbol)
period= "1y"   
interval="1d"

#Get information about the stock's quote type
quote_info = stock.info['quoteType']
st.write(quote_info)

def get_company_name(ticker_symbol):
    company = yf.Ticker(ticker_symbol)
    if is_a_future==False:
        return company.info['longName']
    return

company_name = get_company_name(ticker_symbol)

def get_stock_info(ticker_symbol):
    stock_df = pd.DataFrame() 
    company = yf.Ticker(ticker_symbol)
    stock_info = company.info

    # Creating a DataFrame with stock information
    stock_df = pd.DataFrame({
        'Company Name': [stock_info['longName']],
        'Exchange': [stock_info['exchange']],
        'Sector': [stock_info['sector']],
        'Industry': [stock_info['industry']],
        'Country': [stock_info['country']],
        'Market Cap': [stock_info['marketCap']],
        'Quote Type': [stock_info['quoteType']],
        'Currency': [stock_info['currency']]
    })

    return stock_df


def get_company_exchange(ticker_symbol):
    company = yf.Ticker(ticker_symbol)
    return company.info['exchange']
company_exchange = get_company_exchange(ticker_symbol)


# Function to get data from yahoo  
def get_yahoo_finance_data(ticker_symbol, period, interval):
    try:
        data = yf.download(ticker_symbol, period=period, interval=interval)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return 
        
df = get_yahoo_finance_data(ticker_symbol, period, interval)


# seleccionar si es ACCION O FUTURO  POR QUE MARCA ERROR CON LOS FUTUROS



def get_stock_info(ticker_symbol):
    stock_df = pd.DataFrame() 
    company = yf.Ticker(ticker_symbol)
    stock_info = company.info

    # Creating a DataFrame with stock information
    stock_df = pd.DataFrame({
        'Company Name': [stock_info.get('longName')],
        'Exchange': [stock_info.get('exchange')],
        'Sector': [stock_info.get('sector')],
        'Industry': [stock_info.get('industry')],
        'Country': [stock_info.get('country')],
        'Market Cap': [stock_info.get('marketCap')],
        'Quote Type': [stock_info.get('quoteType')],
        'Currency': [stock_info.get('currency')]
    })

    return stock_df

stock_info = get_stock_info(ticker_symbol)
st.write(stock_info)

sector = stock_info['Sector'].iloc[0]  # Extracting sector information
st.write(sector)

industry = stock_info['Industry'].iloc[0]  # Extracting industry information
st.write(industry)

exchange = stock_info['Exchange'].iloc[0]  # Extracting exchange information

if exchange == "NMS":
    exchange_name = "NASDAQS"
elif exchange == "NYQ":
    exchange_name = "NYSE"
elif exchange == "CME":
    exchange_name = "Chicago Mercantile Exchange"
elif exchange == "NYM":
    exchange_name = "New York Mercantile Exchange"
elif exchange == "CMX":
    exchange_name = "Comex"
elif exchange == "CBT":
    exchange_name = "Chicago Board of Trade"
elif exchange == "WCB":
    exchange_name = "WCB"
    
    




# 2. #####--- GET THE DATA FROM YAHOO FINANCE  ---####

# Calculate the DayRange
df['DayRange'] = df['High'] - df['Low']
# Calculate the percentage range and create a new column 'PercentageRange'
df['PercentageRange'] = (df['DayRange'] / df['Low']) * 100
# Display the DataFrame with the new column


#    #########   H E A D E R S   #####

styled_text = f"""
<div> 
    <h4 style="color: #4287f5;"> Mastering Markets: Unveiling Insights in Stocks and Futures</h4>
    <p style="font-size: 16px; color: #333333;">
        Tap into Market Insights with Numbers! Math and stats power savvy investing in stocks 
        Futures ans Cryptos. Explore these tools to grasp trends, navigate risks, and make informed 
        choices. Dive into data-driven strategies to elevate your investing prowess and confidence."
    </p>
</div>
"""
st.markdown(styled_text, unsafe_allow_html=True)

#  Description
#Dive into market insights using numbers! Discover how math and stats influence 
#stock trends and the ES Future Contract. Uncover the power of data for smarter 
#investing. Explore strategies backed by numbers, turbocharging your investing game 
#and fueling your confidence.

# Display the company name as a header and ticker symbol as a subheader

styled_text = f"""
<div>
    <h4 style="color: #4287f5;">{company_name} . <span style="color: #4287f5;"> ( {ticker_symbol} )   {company_exchange}</span></h4>
    <p style="font-size: 12px; color: #333333;">
        {exchange_name}    {sector}    {industry}
    </p>
</div>
"""
st.markdown(styled_text, unsafe_allow_html=True)


st.write(df.tail(1))


# 3.  Describe statistical values

data = {
    'Metric': ['Mean', 'Median', 'Std Deviation', 'Minimum', 'Maximum', '25th Percentile', 'Variance'],
    'Value': [df['DayRange'].mean(), df['DayRange'].median(), df['DayRange'].std(),
              df['DayRange'].min(), df['DayRange'].max(), df['DayRange'].quantile(0.25), df['DayRange'].var()]
}

# Correlation and Covariance
correlation = df['Adj Close'].corr(df['DayRange'])
covariance = df['Adj Close'].cov(df['DayRange'])

correlation_data = {'Metric': ['Correlation with Adj Close', 'Covariance with Adj Close'], 'Value': [correlation, covariance]}

# Create DataFrames
stats_df = pd.DataFrame(data)
corr_cov_df = pd.DataFrame(correlation_data)

# Display using st.dataframe
st.subheader("Statistical Summary:")
st.dataframe(stats_df)

st.subheader("Correlation and Covariance:")
st.dataframe(corr_cov_df)


# 4. Finds the maximum 'DayRange'
max_day_range_index = df['DayRange'].idxmax()

# Get the row with the maximum 'DayRange'
row_with_max_day_range = df.loc[max_day_range_index]

# Get the date, Open, and Adj Close values
date = row_with_max_day_range.name.strftime('%Y-%m-%d')  # Format date
open_value = "{:,}".format(row_with_max_day_range['Open'])  # Format Open value with comma-separated thousands
adj_close_value = "{:,}".format(row_with_max_day_range['Adj Close'])  # Format Adj Close value with comma-separated thousands

# Display the formatted values using Streamlit
st.write(f"On {date}, Open value: {open_value}, Adj Close value: {adj_close_value}")

# 5 Plots the day range

# *******   CHECAR  *******
# Extract 'DayRange' column data
values = df['DayRange'].values[:200]  # Consider only the first 200 values

# Fit data to a normal distribution
loc, scale = stats.norm.fit(values)

# Create a range of values for the x-axis
x = np.linspace(values.min(), values.max(), len(values))

# Calculate the probability density function (PDF)
param_density = stats.norm.pdf(x, loc=loc, scale=scale)

# Create a figure with two y-axes
fig = go.Figure()

# Add bar chart for histogram
fig.add_trace(go.Bar(
    x=values,
    y=np.histogram(values, bins=30, density=True)[0],
    width=0.1,  # Adjust the width of the bars here
    opacity=0.5,
    name='Histogram',
    marker=dict(color='rgba(0, 0, 255, 0.75)')
))

# Add line chart for PDF with secondary y-axis
fig.add_trace(go.Scatter(
    x=x,
    y=param_density,
    mode='lines',
    name='PDF',
    yaxis='y2',
    line=dict(color='red')
))

# Update layout with axis settings and size adjustments
fig.update_layout(
    title='Sotck Price daily range Histogram ',
    xaxis=dict(title='Predicted Price'),
    yaxis=dict(title='Frequency'),
    yaxis2=dict(title='PDF', overlaying='y', side='right'),
    width=800,  # Adjust the width here
    height=400  # Adjust the height here
)

# Display the combined chart using Streamlit in a TABLE layout


container = st.container()

with container:
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig)

    with col2:

        styled_text = f"""
        <div style="border: 2px solid #4287f5; padding: 10px; border-radius: 5px; background-color: #f0f5f5;">
            <h6 style="color: #4287f5;">The Probability Density Function (PDF)</h6>
            <p style="font-size: 10px; color: #333333;">
            The Probability Density Function (PDF) is a powerful tool used by professional traders, 
            quants, and Artificial Intelligence algorithms. It helps predict potential future prices of stocks or futures 
            contracts. Imagine it as a strategic map showing the likelihood of different price levels within 
            a specific range. This tool empowers you to understand potential outcomes, manage risks effectively, 
            and make more informed investment decisions.
            </p>
        </div>
        """
        st.markdown(styled_text, unsafe_allow_html=True)


# Get the date, Open, and Adj Close values
date = row_with_max_day_range.name.strftime('%Y-%m-%d')  # Format date
open_value = "{:,}".format(row_with_max_day_range['Open'])  # Format Open value with comma-separated thousands
adj_close_value = "{:,}".format(row_with_max_day_range['Adj Close'])  # Format Adj Close value with comma-separated thousands

# Display the formatted values using Streamlit
st.write(f"On {date}, Open value: {open_value}, Adj Close value: {adj_close_value}")

# 5 Plots the day range

# 6 Now lets calculate the daily move  OPEN  vs  CLOSE

# Calculate the daily move based on 'Open', 'High', 'Low', and 'Adj Close'
df['DailyMove'] = df['Adj Close'] - df['Open']

# Identify cases where the 'Close' is higher than the 'Open'
positive_move = df['Close'] - df['Open'] > 0

# Update the 'DailyMove' column for cases where 'Close' is higher than 'Open'
df.loc[positive_move, 'DailyMove'] = df['Close'] - df['Open']

# Display the DataFrame with the new 'DailyMove' column using Streamlit
st.write(df.head(1))

# ------------DailyMove'   PLOTY RED DOTS--------------------



# Calculate the min and max values of 'DailyMove'
min_daily_move = df['DailyMove'].min()
max_daily_move = df['DailyMove'].max()

# Plotting the 'DailyMove' values against dates using a custom color scale
fig = px.scatter(
    df,
    x=df.index,
    y='DailyMove',
    color='DailyMove',
    color_continuous_scale='reds',  # Choose your desired color scale
    labels={'x': '', 'y': 'Daily Move'},
    title='Daily Move vs Date with Gradient Scale'
)

# Styling the color bar and layout
fig.update_traces(marker=dict(size=5, colorscale='Viridis'))  # Adjust colorscale if needed

# Set y-axis limits based on min and max values
fig.update_layout(yaxis=dict(range=[min_daily_move, max_daily_move]))

# Remove color bar label from the right
fig.update_layout(coloraxis_colorbar=dict(title=None))

# Rotating x-axis labels for better readability
#fig.update_layout(xaxis=dict(tickangle=45))

# Show the Plotly figure
st.plotly_chart(fig)


# 5.  Standard deviations
#Plotly to create a normalized bell curve and a bar chart for the frequency of 'Daily Move'
# within 2 standard deviations, combining them into a single plot 


# -----------------BELL CURVE AND HISTOGRAM-----------------------------------------------


# Assuming df is your dataframe

# Calculate mean and standard deviation for the 'DailyMove'
mu, sigma = df['DailyMove'].mean(), df['DailyMove'].std()

# Calculate values within 1 and 2 standard deviations
within_1_std = df[(df['DailyMove'] >= mu - sigma) & (df['DailyMove'] <= mu + sigma)]
within_2_std = df[(df['DailyMove'] >= mu - 2 * sigma) & (df['DailyMove'] <= mu + 2 * sigma)]

# Outliers beyond 2 standard deviations (top and bottom 2.5%)
outliers = df[(df['DailyMove'] < mu - 2 * sigma) | (df['DailyMove'] > mu + 2 * sigma)]

# Number of values within 1 and 2 standard deviations
count_within_1_std = len(within_1_std)
count_within_2_std = len(within_2_std)

# 5% of outliers
count_outliers_5_percent = int(0.05 * len(outliers))

# Extracting top 5% outliers
outliers_5_percent = outliers.head(count_outliers_5_percent)

# Create a range of values for the x-axis within 2 standard deviations
x_within_2_std = np.linspace(mu - 2 * sigma, mu + 2 * sigma, 1000)

# Calculate the normalized bell curve within 2 standard deviations
pdf_within_2_std = norm.pdf(x_within_2_std, mu, sigma)
max_pdf_value = max(pdf_within_2_std)  # Maximum value of the bell curve

#st.title('Normalized Bell Curve and Frequency within 2 Standard Deviations')

# Create subplots with different y-axes for bell curve and frequency
fig, ax1 = plt.subplots(figsize=(8, 6))  # Set the size of the chart (width=8, height=6)

# Plot the bell curve on the first subplot
ax1.plot(x_within_2_std, pdf_within_2_std, color='red', label='Normalized Bell Curve')
ax1.set_xlabel('Daily Move', fontsize=8)  # Adjust X-label font size
ax1.set_ylabel('Probability Density', color='red', fontsize=8)  # Adjust Y-label font size
ax1.tick_params(axis='y', labelcolor='red')
ax1.legend(loc='upper left')

# Create the second subplot sharing the same x-axis as ax1
ax2 = ax1.twinx()

# Calculate the frequency of 'DailyMove' values within 2 standard deviations
bins_within_2_std = np.linspace(mu - 2 * sigma, mu + 2 * sigma, 50)
frequency_within_2_std, _ = np.histogram(df['DailyMove'], bins=bins_within_2_std)

# Set the width of the bars for the frequency bar chart
bar_width = (bins_within_2_std[1] - bins_within_2_std[0]) * 0.8

# Plot the frequency bar chart on the second subplot
ax2.bar(bins_within_2_std[:-1], frequency_within_2_std, width=bar_width, alpha=0.5, color='blue', label='Frequency')
ax2.set_ylabel('Frequency', color='blue', fontsize=8)  # Adjust Y-label font size
ax2.tick_params(axis='y', labelcolor='blue')

# Set y-axis limits separately for both subplots
ax1.set_ylim([0, max_pdf_value])  # Set limits for bell curve subplot
ax2.set_ylim([0, max(frequency_within_2_std)])  # Set limits for frequency subplot

# Show the plot in Streamlit
st.pyplot(fig)

# Additional information if needed
st.write(f"Number of values within 1 standard deviation: {count_within_1_std}")
st.write(f"Number of values within 2 standard deviations: {count_within_2_std}")



#  ------ Create mark down to retrieve information
# 7 # Outliers beyond 2 standard deviations (top and bottom 2.5%)

outliers = df[(df['DailyMove'] < mu - 2 * sigma) | (df['DailyMove'] > mu + 2 * sigma)]

# Display the DataFrame values of the outliers
count_outliers = len(outliers)
#st.write("Total outliers beyond 2 standard deviations:", count_outliers)

# Calculate the percentage of outliers relative to the total number of rows in the DataFrame
percentage_outliers = count_outliers / df.shape[0] * 100

# Display the percentage of outliers
#st.write(f"That represents Percentage of outliers: {percentage_outliers:.2f}%", "   from a total of ", df.shape[0])


#display the statistics regarding the 'DailyMove' column, including counts, 
#ranges within standard deviations, and percentages of outliers 
#  ------ Create mark down to retrieve information

num_rows = len(df)

styled_text = f"""
<div style="border: 2px solid #4287f5; padding: 10px; border-radius: 5px; background-color: #f0f5f5;">
    <h6 style="color: #4287f5;">Statistical daily move of the {ticker_symbol} </h6>
</div>
"""
st.markdown(styled_text, unsafe_allow_html=True)


#----------- container with description  Container with borders and styling


# Calculate mean and standard deviation for the 'DailyMove'
mu, sigma = df['DailyMove'].mean(), df['DailyMove'].std()

# Calculate values within 1 and 2 standard deviations
within_1_std = df[(df['DailyMove'] >= mu - sigma) & (df['DailyMove'] <= mu + sigma)]
within_2_std = df[(df['DailyMove'] >= mu - 2 * sigma) & (df['DailyMove'] <= mu + 2 * sigma)]

# Calculate ranges for positive and negative sides within 1 std deviation
range_positive_1_std = (round(mu, 2), round(mu + sigma, 2))
range_negative_1_std = (round(mu - sigma, 2), round(mu, 2))

# Calculate ranges for positive and negative sides within 2 std deviations
range_positive_2_std = (round(mu, 2), round(mu + 2 * sigma, 2))
range_negative_2_std = (round(mu - 2 * sigma, 2), round(mu, 2))

# Count values in 1 and 2 standard deviations
count_within_1_std = len(within_1_std)
count_within_2_std = len(within_2_std)

# Count positive and negative values within 1 std deviation
positive_within_1_std = len(within_1_std[within_1_std['DailyMove'] > 0])
negative_within_1_std = len(within_1_std[within_1_std['DailyMove'] < 0])

# Count positive and negative values within 2 std deviations
positive_within_2_std = len(within_2_std[within_2_std['DailyMove'] > 0])
negative_within_2_std = len(within_2_std[within_2_std['DailyMove'] < 0])

# Outliers beyond 2 standard deviations (top and bottom 2.5%)
outliers = df[(df['DailyMove'] < mu - 2 * sigma) | (df['DailyMove'] > mu + 2 * sigma)]

# Display the count of outliers and the percentage relative to the total number of rows in the DataFrame
count_outliers = len(outliers)
percentage_outliers = round(count_outliers / df.shape[0] * 100, 2)

# Create a DataFrame with the statistics
data = {
    'Statistic': ['Values in 1 std dev', 'Range positive trend', 'Range negative trend',
                  'SValues in 2 std dev', 'Range positive trend Positive 2std', 'Range negative trend 2std)',
                  'Count of positive within 1 std', 'Count of negative within 1 std',
                  'Count of positive within 2 std', 'Count of negative within 2 std',
                  'Percentage of outliers'],
    'Value': [count_within_1_std, range_positive_1_std, range_negative_1_std,
              count_within_2_std, range_positive_2_std, range_negative_2_std,
              positive_within_1_std, negative_within_1_std,
              positive_within_2_std, negative_within_2_std,
              f'{percentage_outliers}%']
}

# Assigning the second value of range_positive_1_std to range_upper variable
range_upper = range_positive_1_std[1]
range_down = range_negative_1_std[0]
range_upper2 = range_positive_2_std[1]
range_down2 = range_negative_2_std[0]


container = st.container()

with container:
    col1, col2 = st.columns(2)

    with col1:
        statistics_df = pd.DataFrame(data)
        #st.write("Statistics on DailyMove:")
        st.write(statistics_df)

    with col2:

        styled_text = f"""
            <div style="border: 2px solid #4287f5; padding: 10px; border-radius: 5px; background-color: #f0f5f5;">
            <h6 style="color: #4287f5;">Statistical daily move of the {ticker_symbol}</h6>
            <p style="font-size: 12px; color: #333333;">Range within <strong>"One Positive Standard Deviation"</strong>is your safety zone—it tells us that roughly 70% of the time, 
            the stock remains within this range. Anticipating it to shoot higher is like betting against a 30% chance. 
            On the other hand, 'Positive and Negative Range within One standard deviation gets us ready for the ups and downs—it's 
            a signal for potential swings. These ranges act like your strategic guide and anticipate the range of price movements for smarter trades.
            <br>
            <br>
            For <strong>{ticker_symbol}, about 70% of the time during an uptrend</strong>, the asset typically won't exceed <strong>{range_upper}</strong>. 
            Conversely, when the market is downtrending, it tends to stay above <strong> {range_down} </strong>, again around 70% of the time.
            <br>
            <br>
            Also in approximately <strong> 95% </strong> of situations, the asset's movement stays within a broader scope— <strong> {range_upper2} </strong> for 
            upward movements and <strong> {range_down2} </strong>for downward movements. These figures play a crucial role in trades, guiding decisions 
            like setting your 'stop loss' or determining your 'target' price.
            </p>
        </div>
        """

        styled_text = styled_text.replace('\n', '')  # Remove the newline characters
        st.markdown(styled_text, unsafe_allow_html=True)    




# ------ RANGES  -Create a DataFrame with 'low_range' and 'high_range' columns

#These rangers apply to a FUTURE MARKET TICKET LIKE ES or NQ
data = {
    'low_range': list(range(300)),
    'high_range': [x + 0.75 for x in range(300)]
}

new_df = pd.DataFrame(data)

# Function to count frequency within 'low_range' and 'high_range' in 'df' DataFrame
def count_frequency(row):
    low = row['low_range']
    high = row['high_range']
    frequency = len(df[(df['DayRange'] >= low) & (df['DayRange'] <= high)])
    return frequency

# Apply the function to each row of 'new_df' DataFrame to calculate 'frequency'
new_df['frequency'] = new_df.apply(count_frequency, axis=1)

# Create 'frequency_plot' column for charting purposes
new_df['frequency_plot'] = new_df['frequency'].head(150)

# Calculate sum of the 'frequency' column
frequency_sum = new_df['frequency'].sum()
new_df['percentage'] = (new_df['frequency'] / frequency_sum)
new_df['cum%'] = new_df['percentage'].cumsum()



#   CHECK LAYOUT   MEJORAR   ********************************



# Display statistical information for ES DAILY RANGE
mean_dayrange = round(df['DayRange'].mean(), 2)
median_dayrange = round(df['DayRange'].median(), 2)

st.write(f"Mean DayRange: {mean_dayrange}")
st.write(f"Median DayRange: {median_dayrange}")




# Get the maximum value from 'frequency_plot'
max_frequency_plot = new_df['frequency_plot'].max()
st.write(f"Maximum value from 'frequency_plot': {max_frequency_plot}")


st.write (new_df['frequency_plot'])

# Plot 'frequency_plot' using Plotly
fig = px.bar(new_df, x=new_df.index, y='frequency_plot', labels={'x': 'Index', 'frequency_plot': 'Frequency Plot'},
             title='Frequency Plot')

# Define the range you want to display on the x-axis (adjust these values according to your data)
x_min = 0  # Replace with your desired minimum value
x_max = 10  # Replace with your desired maximum value

# Adjust x-axis range
fig.update_xaxes(range=[x_min, x_max])  # Set the x-axis range to display specific values

st.plotly_chart(fig)



# Get the maximum value from 'frequency_plot'
max_frequency_plot = new_df['frequency_plot'].max()
st.write(f"Maximum value from 'frequency_plot': {max_frequency_plot}")


# Plot 'frequency_plot' using Plotly
fig = px.bar(new_df, x=new_df.index, y='frequency_plot', labels={'x': 'Index', 'frequency_plot': 'Frequency Plot'},
             title='Frequency Plot')
st.plotly_chart(fig)


#  MARKDOWN TEXT  (BETTER READING OF DATA)

styled_text = f"""
<div style="border: 2px solid #4287f5; padding: 10px; border-radius: 5px; background-color: #f0f5f5;">
    <h6 style="color: #4287f5;"> {ticker_symbol} Mean dily range</h6>
    <p style="font-size: 10px; color: #333333;">
        Mean DayRange: {mean_dayrange}
        Maximum value from 'frequency_plot': {max_frequency_plot}"
    </p>
</div>
"""
st.markdown(styled_text, unsafe_allow_html=True)


#..-....... OTHER   Create a bar chart

fig = go.Figure()

# Adding bars to the figure
fig.add_trace(go.Bar(
    y=new_df['low_range'].iloc[:80],
    x=new_df['frequency_plot'].iloc[:80],
    orientation='h',
    marker=dict(color='blue')
))

# Get the indices where condition is met
indices = []
for i in range(len(new_df['cum%'])):
    if new_df['cum%'].iloc[i] >= 0.68 and new_df['cum%'].iloc[i - 1] < 0.68:
        indices.append(i)

# Update the color of specific bars
fig.update_traces(marker=dict(color='blue'), selector=dict(marker_color='blue'))  # Set all bars to blue
fig.update_traces(marker=dict(color='orange'), selector=dict(marker_color='orange', row=indices))  # Change specific bars to orange

# Update layout
fig.update_layout(
    #title='E-mini',
    xaxis=dict(title='E-mini S&P 500 futures daily range'),
    yaxis=dict(title='Session Range'),
    yaxis_categoryorder='total ascending',  # This line keeps the descending order of the bars
    width=300,  # Set width to 300 pixels
    height=650  # Set height to 500 pixels
)

st.plotly_chart(fig)


col1, col2= st.columns(2)

with col1:
        st.plotly_chart(fig)
        
with col2:
   
   st.markdown(
    """
    <div style="border: 2px solid #4287f5; padding: 10px; border-radius: 5px; background-color: #f0f5f5;">
        <h6 style="color: #4287f5;">Understanding the Instrument You Trade</h6>
        <p style="font-size: 10px; color: #333333;">
            Delving into the instrument you're trading, especially regarding statistical data,
            is vital. Micro E-mini Futures encompass crucial technical values, pivotal for 
            crafting a winning trading strategy. This encompasses grasping the current market 
            condition (Market Internals), understanding market structure (Market Profile), 
            navigating bid dynamics (Reading the tape), and utilizing statistical figures 
            to gauge ranges based on previous Highs, Lows, and Opening Balances. These stats 
            provide foresight into potential moves or corrections in Micro E-mini Futures. 
            While the market evolves, it often adheres to anticipated patterns, crucial 
            to recognize for successful trading.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


#8 .   interactive plot of the normalized probability density function and the histogram for the 'DayRange' data. 

# Extract 'DayRange' column data
values = df['DayRange'].values[:100]  # Consider only the first 100 values

# Fit data to a normal distribution
loc, scale = stats.norm.fit(values)

# Create a range of values for the x-axis
x = np.linspace(values.min(), values.max(), len(values))

# Calculate the probability density function (PDF)
param_density = stats.norm.pdf(x, loc=loc, scale=scale)
label = f"mean={loc:.4f}, std={scale:.4f}"

# Create subplots
fig = go.Figure()

# Add histogram subplot
fig.add_trace(go.Histogram(x=values, nbinsx=30, name='Histogram', opacity=0.5,
                           histnorm='probability density', marker_color='blue'))

# Add PDF subplot
fig.add_trace(go.Scatter(x=x, y=param_density, mode='lines', name='PDF', line=dict(color='red')))

# Update layout and show plot
fig.update_layout(title='Normalized Probability Density Function and Histogram',
                  xaxis=dict(title='DayRange'),
                  yaxis=dict(title='Frequency', overlaying='y', side='left'),
                  yaxis2=dict(title='PDF', overlaying='y', side='right', color='red'),
                  legend=dict(y=1.0, traceorder='normal'))
st.plotly_chart(fig)



#----- Sumarize data --------------------------

#define the categories to determinate the range

st.write ("base dataframe")
st.write(df.tail(1))

# Get the maximum value of the 'DayRange' column
max_day_range = df['DayRange'].max()
st.write("Max day range", max_day_range)

# Define the maximum value from the 'DayRange' column as an integer
max_value = int(max_day_range)

# Define 10 bins for values ranging from 0 to the maximum value in 'DayRange'
if max_value >= 10:
    bins_10 = list(range(0, max_value + 1, max_value // 10))
else:
    bins_10 = [i for i in range(0, max_value + 1)]

# Categorize values into bins and count occurrences
df['Category'] = pd.cut(df['DayRange'], bins=bins_10, right=False)
category_counts = df['Category'].value_counts().sort_index().reset_index()
category_counts.columns = ['Category', 'Frequency']

# Convert Interval objects to strings for Streamlit visualization
category_counts['Category'] = category_counts['Category'].astype(str)

# Create an interactive histogram using Plotly Express
fig = px.bar(category_counts, x='Category', y='Frequency', labels={'Frequency': 'Frequency Count'})
fig.update_traces(marker_color='skyblue')

# Streamlit - Display the Plotly figure
st.plotly_chart(fig)



#PIE CHART   REVISED  10/01/2024


# Assuming you have a DataFrame named category_counts containing your calculated data

# Create a pie chart using Plotly Express
fig = px.pie(category_counts, values='Frequency', names='Category')

# Set layout parameters
fig.update_traces(textinfo='percent+label', pull=[0.05] * len(category_counts))  # Add percentage values to each slice

# Formatting percentages with 0 decimals
fig.update_traces(textinfo='percent', textfont_size=12, textposition='inside', hoverinfo='percent+label')

col1, col2 = st.columns(2)

with col1:
    st.write(" Display the frequency of occurrences for each category")
    st.write(category_counts)

with col2:
    # Display the pie chart using Streamlit with specific pixel sizes
    st.plotly_chart(fig, use_container_width=True, width=300, height=300)
    
    


#-------------below past version delete maybe


# Create a DataFrame with 'low_range', 'high_range', and 'labels' columns
data = {
    'low_range': [0, 11, 21, 31, 41],
    'high_range': [10, 20, 30, 40, df['DayRange'].max()],
    'labels': ['0 - 10 Handles', '11 - 20 Handles', '21 - 30 Handles', '31 - 40 Handles', '41 - all']
}

# Create a DataFrame
new_df = pd.DataFrame(data)

# Sort 'high_range' values in ascending order
new_df.sort_values('high_range', inplace=True)

# Create a new column in 'df' that categorizes 'DayRange' values based on the ranges in 'new_df'
df['Range_Category'] = pd.cut(df['DayRange'], bins=new_df['high_range'])

# Aggregate the data in 'df' based on the new ranges and calculate the mean
aggregated_data = df.groupby('Range_Category')['DayRange'].mean().reset_index()

# Display the aggregated data using Streamlit
#st.write(aggregated_data)




# Function to count frequency within 'low_range' and 'high_range' in 'df' DataFrame
def count_frequency(row):
    low = row['low_range']
    high = row['high_range']
    frequency = len(df[(df['DayRange'] >= low) & (df['DayRange'] <= high)])
    return frequency

# Apply the function to each row of 'new_df' DataFrame to calculate 'frequency'
new_df['frequency'] = new_df.apply(count_frequency, axis=1)

# Calculate total frequency sum
frequency_sum = new_df['frequency'].sum()

# Calculate percentage and cumulative percentage
new_df['percentage'] = new_df['frequency'] / frequency_sum
new_df['cum'] = new_df['percentage'].cumsum()

# Displaying the statistical information for 'DayRange' column in 'df'
mean_dayrange = round(df['DayRange'].mean(), 2)
median_dayrange = round(df['DayRange'].median(), 2)

st.write(f"Mean DayRange: {mean_dayrange}")
st.write(f"Median DayRange: {median_dayrange}")


# Export the data to Excel (optional)
#new_df.to_excel('new_df_data.xlsx', index=False)

# Values in Dollars per tick and day range  

#lista de futuros y sus valores

data_futures_df = pd.DataFrame(data_futures)
# Formatting 'Pertic$' column to US dollars with 2 decimal places
data_futures_df['Pertic$'] = data_futures_df['Pertic$'].map('${:,.2f}'.format)


st.write (data_futures_df)


# Conditionally retrieve rows based on is_a_future and Symbol
if is_a_future:
    query_result = data_futures_df[data_futures_df['Symbol'] == ticker_symbol]
    st.write(query_result)
else:
    st.write("Not a future.")




#For the stock I need to chang some parameters dinamically  example aapl  the range is below 7 usd  



