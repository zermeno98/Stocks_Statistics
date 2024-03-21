# Streamlit Stock and Futures Analysis Tool

This repository contains a Streamlit implementation designed for analyzing both stocks and futures. The tool provides users with a user-friendly interface for conducting comprehensive statistical analyses on financial data, empowering them to make informed decisions in trading and investment.

## Global Variables

The code includes global variables for defining the ticker symbol and whether the selected asset is a future or not. It initializes with a default ticker symbol "AAPL" and sets the flag `is_a_future` to `False`.

## Data Structures

The code defines two dataframes: one for futures contracts and another for stocks. The futures contracts dataframe includes information such as symbol, name, class, tick, and price tick. The stocks dataframe contains symbol and name for various stocks.

## Functionality

The `select_ticker()` function allows users to select between futures contracts and stocks using Streamlit's sidebar. Depending on the selection, users can choose assets either from a list or by entering a new ticker symbol.

The code fetches data from Yahoo Finance based on the selected ticker symbol, period, and interval. It also provides functionalities to extract company information, including name, exchange, sector, industry, country, market cap, quote type, and currency.

## Display and Visualization

The tool generates an interactive display using Streamlit, showcasing selected ticker symbol, company name, exchange, sector, industry, and other relevant information. Additionally, it calculates and displays the DayRange and PercentageRange for the selected asset.

## Usage

To use this tool, clone the repository and install the required dependencies. Run the Streamlit application using `streamlit run app.py` command. Access the interface in your web browser, select the asset category, and input the desired parameters to explore statistical insights and visualizations.

## Contributions

Contributions to this project are welcome! If you have any suggestions, feature requests, or bug fixes, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
