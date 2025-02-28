# Forecasting Tool for Swing Trading

This application is designed to enhance swing trading strategies by providing valuable insights through accurate stock price forecasts and various analytical tools.

## Features

- **Custom-Tuned Prophet Forecasting Model**: Achieves a median SMAPE of 15% across 150 diverse stock tickers, demonstrating robust predictive capability.
- **Volatility-Adjusted Winsorization**: Effectively mitigates the impact of outliers, contributing to the model's overall robustness, especially in high-volatility scenarios.
- **Dynamic Training Data**: Adjusts based on stock volatility to improve prediction accuracy.
- **Visual Aids**: Includes candlestick charts, Bollinger Bands, and Simple Moving Averages (SMAs) to help identify potential entry and exit points.
- **Statistical Significance**: Validated through a Wilcoxon signed-rank test (p < 0.05) and a large effect size (Cliff's Delta = 0.69) compared to the standard Prophet model.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/shanejp76/Forecasting-Tool-for-Swing-Trading.git](https://www.google.com/search?q=https://github.com/shanejp76/Forecasting-Tool-for-Swing-Trading.git)
   cd Forecasting-Tool-for-Swing-Trading
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate      # On Windows
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

Documentation
For detailed information about the model, methodology, and performance, please refer to the [Forecasting Tool Documentation](https://github.com/shanejp76/Forecasting-Tool-for-Swing-Trading/blob/docs/Forecasting%20Tool%20Documentation.pdf).

### About

#### Swing Trading

This application was designed to enhance my swing trading strategy. Swing trading focuses on capturing short-term price movements, and this tool provides accurate forecasting data.

I've integrated a Prophet forecasting model, fine-tuned with techniques like winsorization and hyperparameter tuning to optimize its accuracy. The model dynamically adjusts its training data based on the stock's volatility, aiming to improve prediction accuracy.

By combining these analytical tools with visual aids like candlestick charts, Bollinger Bands, and SMAs, I'm able to identify potential entry and exit points with greater confidence, ultimately refining my trading decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
