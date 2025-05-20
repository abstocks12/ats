# Automated Trading System

A comprehensive system for automated trading with modular components for data collection, analysis, prediction, and execution.

## Features

- **Portfolio Management**: Dynamically add and remove instruments
- **Data Collection**: Market data, news, financial data, and global indicators
- **Analysis Engine**: Technical, fundamental, and statistical analysis
- **Machine Learning**: Daily predictions and trading opportunity detection
- **Real-Time Trading**: Automated execution with risk management
- **Reporting**: Daily reports and WhatsApp notifications

## Requirements

- Python 3.10+
- MongoDB
- Zerodha Kite Connect API

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure settings
6. Initialize the database: `python main.py --init-db`

## Usage

### Add an instrument to the portfolio

```bash
python main.py --add-instrument SYMBOL:EXCHANGE:TYPE


# Running Scripts in the Automated Trading System

Here are the commands to run the various scripts in your automated trading system, with examples using Python 3. These are organized by typical workflow order and use case.

## Basic System Operation

### 1. Run the Main System

To run the main system in paper trading mode:
```bash
python3 main.py --mode paper
```

To run in live trading mode:
```bash
python3 main.py --mode live
```

To run with specific instruments:
```bash
python3 main.py --mode paper --instruments TATASTEEL:NSE:equity,RELIANCE:NSE:equity
```

To run in backtest mode with date range:
```bash
python3 main.py --mode backtest --backtest-start 2024-01-01 --backtest-end 2024-04-30
```

### 2. Zerodha Authentication

To get the Zerodha login URL (first step):
```bash
python3 scripts/zerodha_login.py
```

After visiting the URL and getting the request token, authenticate:
```bash
python3 scripts/zerodha_login.py --request-token YOUR_REQUEST_TOKEN_HERE
```

## Portfolio Management

### 3. Add Instruments

To add a single instrument:
```bash
python3 scripts/add_instrument.py --symbol INFY --exchange NSE --type equity
```

To add an instrument with additional parameters:
```bash
python3 scripts/add_instrument.py --symbol INFY --exchange NSE --type equity --sector technology --position-size 5 --max-risk 1
```

### 4. Remove Instruments

To remove an instrument:
```bash
python3 scripts/remove_instrument.py --symbol INFY --exchange NSE
```

To force removal even with open positions:
```bash
python3 scripts/remove_instrument.py --symbol INFY --exchange NSE --force
```

## Data Collection and Analysis

### 5. Collect Market Data

To collect historical data for all instruments:
```bash
python3 scripts/collect_data.py --data-type historical
```

To collect data for a specific instrument:
```bash
python3 scripts/collect_data.py --data-type historical --symbol TATASTEEL --exchange NSE
```

To collect financial data:
```bash
python3 scripts/collect_data.py --data-type financial
```

To collect news data:
```bash
python3 scripts/collect_data.py --data-type news
```

### 6. Generate Predictions

To generate predictions for all instruments:
```bash
python3 scripts/generate_predictions.py
```

For a specific instrument:
```bash
python3 scripts/generate_predictions.py --symbol TATASTEEL --exchange NSE
```

## Trade Control

### 7. Start/Stop Trading

To start trading manually:
```bash
python3 scripts/start_trading.py
```

To start trading with specific instruments:
```bash
python3 scripts/start_trading.py --instruments TATASTEEL:NSE,RELIANCE:NSE
```

To stop trading:
```bash
python3 scripts/stop_trading.py
```

To stop trading and close all positions:
```bash
python3 scripts/stop_trading.py --close-positions
```

## Reports and Monitoring

### 8. Generate Reports

To generate the morning report:
```bash
python3 scripts/send_reports.py --report-type morning
```

To generate the end-of-day report:
```bash
python3 scripts/send_reports.py --report-type eod
```

To generate a weekly report:
```bash
python3 scripts/send_reports.py --report-type weekly
```

### 9. System Maintenance

To back up the database:
```bash
python3 scripts/backup_database.py
```

## Debugging and Development

### 10. Database Operations

To initialize the database schema:
```bash
python3 scripts/initialize_database.py
```

To optimize the database:
```bash
python3 scripts/optimize_database.py
```

### 11. Testing Components

To test a specific strategy:
```bash
python3 scripts/test_strategy.py --strategy SMA_Crossover --symbol TATASTEEL --exchange NSE
```

To test the notification system:
```bash
python3 scripts/test_notification.py --message "Test notification" --channel whatsapp
```

## Notes on Usage

1. Most scripts accept a `--debug` flag for verbose output:
   ```bash
   python3 main.py --mode paper --debug
   ```

2. The log level can be adjusted:
   ```bash
   python3 main.py --log-level DEBUG
   ```

3. For scripts not listed above, you can usually get help with:
   ```bash
   python3 scripts/script_name.py --help
   ```

These commands should cover most of the typical operations you'll need to perform with your automated trading system. Remember to configure your environment variables properly in the `.env` file before running any scripts.