# Forecasting Tool for Swing Trading

This application is designed to enhance swing trading strategies by providing valuable insights through various analytical tools and visual aids.

## Features

- **Prophet Forecasting Model**: Integrated with techniques like winsorization and hyperparameter tuning to optimize accuracy.
- **Dynamic Training Data**: Adjusts based on stock volatility to improve prediction accuracy.
- **Visual Aids**: Includes candlestick charts, Bollinger Bands, and SMAs to help identify potential entry and exit points.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/swing-ticker.git
    cd swing-ticker
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```bash
    streamlit run main.py
    ```

2. Follow the instructions on the Streamlit interface to interact with the application.

## Appendix

### Ticker List

Displays the list of tickers used in the analysis.

### Raw Data

Shows the raw data used for forecasting.

### Forecast Grid

Displays the forecasted data.

### Forecast Components

Shows the components of the forecast.

### Model Iterations

Displays the accuracy metrics for different models:
- Baseline Model
- Winsorized Model
- Final Model

### About

#### Swing Trading

This application was designed to enhance my swing trading strategy. Swing trading focuses on capturing short-term price movements, and this tool provides me with valuable insights.

I've integrated a Prophet forecasting model, fine-tuned with techniques like winsorization and hyperparameter tuning to optimize its accuracy. The model dynamically adjusts its training data based on the stock's volatility, aiming to improve prediction accuracy.

By combining these analytical tools with visual aids like candlestick charts, Bollinger Bands, and SMAs, I'm able to identify potential entry and exit points with greater confidence, ultimately refining my trading decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
