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