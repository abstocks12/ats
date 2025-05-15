"""
Portfolio-specific configuration settings.
Defines trading parameters for different instrument types and sectors.
"""

# Default trading parameters by instrument type
DEFAULT_TRADING_PARAMS = {
    'equity': {
        'position_size_percent': 5.0,  # % of capital per position
        'max_risk_percent': 1.0,       # % of capital at risk per position
        'stop_loss_percent': 2.0,      # Default stop loss %
        'target_percent': 6.0,         # Default target %
        'default_timeframe': 'intraday',
        'strategies': ['technical', 'event_driven'],
        'trailing_stop': True,
        'partial_booking': True
    },
    'futures': {
        'position_size_percent': 3.0,
        'max_risk_percent': 0.8,
        'stop_loss_percent': 1.5,
        'target_percent': 4.5,
        'default_timeframe': 'swing',  # 1-5 days
        'strategies': ['technical', 'statistical_arbitrage'],
        'trailing_stop': True,
        'partial_booking': True
    },
    'options': {
        'position_size_percent': 2.0,
        'max_risk_percent': 0.5,
        'stop_loss_percent': 20.0,     # Higher for options due to volatility
        'target_percent': 50.0,
        'default_timeframe': 'intraday',
        'strategies': ['volatility', 'event_driven'],
        'trailing_stop': False,
        'partial_booking': True
    }
}

# Sector-specific overrides
SECTOR_PARAMS = {
    'banking': {
        'equity': {
            'position_size_percent': 4.0,
            'stop_loss_percent': 2.5,
            'strategies': ['technical', 'event_driven', 'fundamental']
        }
    },
    'it': {
        'equity': {
            'position_size_percent': 6.0,
            'stop_loss_percent': 3.0,
            'strategies': ['technical', 'trend_following']
        }
    },
    'pharma': {
        'equity': {
            'position_size_percent': 3.0,
            'max_risk_percent': 0.8,
            'strategies': ['technical', 'event_driven', 'fundamental']
        }
    },
    'auto': {
        'equity': {
            'position_size_percent': 4.5,
            'strategies': ['technical', 'fundamental']
        }
    },
    'fmcg': {
        'equity': {
            'position_size_percent': 5.0,
            'stop_loss_percent': 1.5,
            'strategies': ['fundamental', 'trend_following']
        }
    }
}

# Strategy-specific parameters
STRATEGY_PARAMS = {
    'technical': {
        'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger'],
        'lookback_periods': [20, 50, 200],
        'signal_confirmation_count': 2,  # Number of signals needed for confirmation
        'min_volume_percentile': 50      # Minimum volume percentile for signals
    },
    'fundamental': {
        'min_roe': 15.0,
        'max_pe': 30.0,
        'min_profit_growth': 10.0,
        'max_debt_equity': 1.0,
        'consider_dividend': True
    },
    'event_driven': {
        'pre_event_days': 3,
        'post_event_days': 2,
        'min_sentiment_score': 0.6,
        'news_volume_threshold': 3  # Minimum number of news articles to consider
    },
    'statistical_arbitrage': {
        'correlation_threshold': 0.7,
        'z_score_entry': 2.0,
        'z_score_exit': 0.5,
        'max_holding_days': 5,
        'pair_rebalance_days': 30
    },
    'trend_following': {
        'min_adx': 25,
        'trend_confirmation_days': 3,
        'pullback_entry': True,
        'momentum_threshold': 0.8
    },
    'volatility': {
        'iv_percentile_threshold': 70,
        'vix_consideration': True,
        'skew_analysis': True,
        'mean_reversion_bias': True
    }
}

# Risk management parameters
RISK_MANAGEMENT = {
    'max_portfolio_risk': 5.0,  # Maximum % of capital at risk across all positions
    'max_sector_exposure': 25.0,  # Maximum % of capital in a single sector
    'max_instrument_exposure': 10.0,  # Maximum % of capital in a single instrument
    'correlation_risk_adjustment': True,  # Adjust position size based on correlation
    'market_regime_adjustment': True,  # Adjust risk based on market regime
    'volatility_adjustment': True,  # Adjust position size based on volatility
    'drawdown_protection': {
        'account_drawdown_threshold': 5.0,  # % drawdown to trigger protection
        'strategy_drawdown_threshold': 10.0,  # % drawdown for a strategy
        'position_size_reduction': 50.0,  # % reduction in position size
        'temporary_stop_threshold': 15.0  # % drawdown to temporarily stop trading
    }
}

# Timeframe definitions (in minutes except for 'day' and 'week')
TIMEFRAMES = {
    'tick': 1,
    '1min': 1,
    '3min': 3,
    '5min': 5, 
    '15min': 15,
    '30min': 30,
    '60min': 60,
    'day': 'day',
    'week': 'week'
}

# Trading time classifications
TRADING_TIMEFRAMES = {
    'scalping': {
        'max_holding_time': 30,  # minutes
        'typical_targets': [0.5, 1.0],  # %
        'typical_stops': [0.3, 0.5]  # %
    },
    'intraday': {
        'max_holding_time': 'day',
        'typical_targets': [1.0, 3.0],
        'typical_stops': [0.5, 1.5]
    },
    'swing': {
        'max_holding_time': 5,  # days
        'typical_targets': [3.0, 10.0],
        'typical_stops': [1.5, 5.0]
    },
    'positional': {
        'max_holding_time': 30,  # days
        'typical_targets': [10.0, 25.0],
        'typical_stops': [5.0, 10.0]
    },
    'long_term': {
        'max_holding_time': 365,  # days
        'typical_targets': [25.0, 100.0],
        'typical_stops': [10.0, 20.0]
    }
}

def get_instrument_params(instrument_type, sector=None):
    """
    Get trading parameters for a specific instrument type and sector
    
    Args:
        instrument_type (str): Type of instrument ('equity', 'futures', 'options')
        sector (str, optional): Sector of the instrument
        
    Returns:
        dict: Dictionary of trading parameters
    """
    # Start with default parameters for the instrument type
    if instrument_type not in DEFAULT_TRADING_PARAMS:
        raise ValueError(f"Unknown instrument type: {instrument_type}")
    
    params = DEFAULT_TRADING_PARAMS[instrument_type].copy()
    
    # Apply sector-specific overrides if available
    if sector and sector in SECTOR_PARAMS:
        sector_overrides = SECTOR_PARAMS[sector].get(instrument_type, {})
        params.update(sector_overrides)
    
    return params

def get_strategy_params(strategy_name):
    """
    Get parameters for a specific trading strategy
    
    Args:
        strategy_name (str): Name of the strategy
        
    Returns:
        dict: Dictionary of strategy parameters
    """
    if strategy_name not in STRATEGY_PARAMS:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return STRATEGY_PARAMS[strategy_name].copy()