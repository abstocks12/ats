# options.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import math

class OptionsTrader:
    def __init__(self, db_connector):
        """Initialize the options trader"""
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
    
    def analyze_trading_opportunity(self, symbol, exchange, expiry=None, strategy=None, 
                                  prediction=None, timeframe="day"):
        """
        Analyze potential trading opportunities for options
        
        Parameters:
        - symbol: Trading symbol (underlying)
        - exchange: Exchange (e.g., NSE)
        - expiry: Optional specific expiry date to analyze
        - strategy: Optional specific option strategy to analyze
        - prediction: Optional prediction data for the underlying
        - timeframe: Timeframe for analysis
        
        Returns:
        - Dictionary with trading opportunity details
        """
        try:
            # Get available options for this symbol
            options = self._get_options(symbol, exchange, expiry)
            if not options or not options.get("calls") or not options.get("puts"):
                return {"status": "error", "message": "No options available"}
            
            # If no specific expiry, use the nearest one
            if not expiry:
                expiry = options["expiries"][0]  # Nearest expiry
            
            # Get underlying data
            underlying_data = self._get_underlying_data(symbol, exchange, timeframe)
            if not underlying_data or len(underlying_data) < 20:
                return {"status": "error", "message": "Insufficient underlying data"}
            
            # Current price of underlying
            current_price = underlying_data["close"].iloc[-1]
            
            # Get technical analysis for underlying
            technical = self._get_technical_analysis(symbol, exchange, timeframe)
            
            # Get implied volatility surface and skew
            iv_data = self._analyze_iv_surface(options, current_price)
            
            # Get market context
            market_context = self._get_market_context(symbol, exchange)
            
            # Analyze option greeks and metrics
            option_metrics = self._analyze_option_metrics(options, current_price)
            
            # Analyze option volume and open interest
            volume_oi = self._analyze_volume_oi(options)
            
            # Identify appropriate strategy if not specified
            if not strategy:
                strategy = self._determine_best_strategy(
                    prediction, technical, iv_data, 
                    market_context, option_metrics, 
                    current_price, expiry
                )
            
            # Get specific strategy parameters
            strategy_params = self._get_strategy_parameters(
                strategy, symbol, exchange, options, 
                current_price, expiry, technical, 
                iv_data, market_context, option_metrics
            )
            
            # If no suitable parameters found
            if not strategy_params.get("suitable_for_trading", False):
                return {
                    "status": "success",
                    "trade_opportunity": False,
                    "reason": strategy_params.get("reason", "No suitable parameters found")
                }
            
            # Construct trading opportunity
            opportunity = {
                "status": "success",
                "symbol": symbol,
                "exchange": exchange,
                "expiry": expiry,
                "days_to_expiry": (expiry - datetime.now()).days,
                "timestamp": datetime.now(),
                "trade_opportunity": True,
                "underlying_price": current_price,
                "strategy": strategy,
                "strategy_params": strategy_params,
                "leg1": strategy_params.get("leg1"),
                "leg2": strategy_params.get("leg2"),
                "leg3": strategy_params.get("leg3", None),
                "leg4": strategy_params.get("leg4", None),
                "max_profit": strategy_params.get("max_profit"),
                "max_loss": strategy_params.get("max_loss"),
                "break_even": strategy_params.get("break_even"),
                "profit_probability": strategy_params.get("profit_probability"),
                "risk_reward": strategy_params.get("risk_reward"),
                "implied_volatility": iv_data.get("atm_iv"),
                "iv_percentile": iv_data.get("iv_percentile"),
                "technical_signals": technical["signals"] if technical else None,
                "market_context": market_context["summary"] if market_context else None
            }
            
            # Save opportunity to database
            self._save_opportunity(opportunity)
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error analyzing options opportunity: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_options(self, symbol, exchange, expiry=None):
        """Get available options for a symbol"""
        try:
            # Base query for options
            query = {
                "underlying": symbol,
                "exchange": exchange,
                "instrument_type": "option",
                "status": "active"
            }
            
            # Add expiry filter if provided
            if expiry:
                query["expiry"] = expiry
            
            # Get all options
            options = list(self.db.instrument_collection.find(query))
            
            if not options:
                return None
            
            # Group by expiry
            expiries = set()
            calls = {}
            puts = {}
            
            for option in options:
                exp = option["expiry"]
                expiries.add(exp)
                
                if exp not in calls:
                    calls[exp] = []
                if exp not in puts:
                    puts[exp] = []
                
                if option["option_type"] == "call":
                    calls[exp].append(option)
                else:
                    puts[exp].append(option)
            
            # Sort expiries
            sorted_expiries = sorted(expiries)
            
            # Get current market data for each option
            for exp in sorted_expiries:
                for call in calls.get(exp, []):
                    call["market_data"] = self._get_option_data(call["_id"])
                
                for put in puts.get(exp, []):
                    put["market_data"] = self._get_option_data(put["_id"])
            
            return {
                "expiries": sorted_expiries,
                "calls": calls,
                "puts": puts
            }
            
        except Exception as e:
            self.logger.error(f"Error getting options: {str(e)}")
            return None
    
    def _get_option_data(self, option_id):
        """Get latest market data for an option"""
        try:
            # Query for option data
            query = {
                "instrument_id": option_id
            }
            
            # Get the latest data point
            data = self.db.market_data_collection.find_one(
                query,
                {"timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, 
                 "open_interest": 1, "iv": 1, "delta": 1, "gamma": 1, "theta": 1, "vega": 1}
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting option data: {str(e)}")
            return None
    
    def _get_underlying_data(self, symbol, exchange, timeframe):
        """Get market data for the underlying"""
        try:
            # Query for underlying data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": "equity",
                "timeframe": timeframe
            }
            
            # Get the last 100 data points
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            ).sort("timestamp", -1).limit(100))
            
            if not data:
                return None
                
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting underlying data: {str(e)}")
            return None
    
    def _get_technical_analysis(self, symbol, exchange, timeframe):
        """Get technical analysis for underlying"""
        try:
            # Try to get from TechnicalAnalyzer
            try:
                from research.technical_analyzer import TechnicalAnalyzer
                analyzer = TechnicalAnalyzer(self.db)
                analysis = analyzer.analyze(symbol, exchange, timeframe)
                return analysis
            except Exception as tech_error:
                self.logger.warning(f"Technical analyzer not available: {str(tech_error)}")
            
            # Fallback to database query
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "analysis_type": "technical"
            }
            
            analysis = self.db.analysis_collection.find_one(query)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error getting technical analysis: {str(e)}")
            return None
    
    def _get_market_context(self, symbol, exchange):
        """Get market context and sentiment analysis"""
        try:
            # Try to get market analysis
            try:
                from research.market_analysis import MarketAnalyzer
                analyzer = MarketAnalyzer(self.db)
                analysis = analyzer.get_market_context(exchange)
                return analysis
            except Exception as market_error:
                self.logger.warning(f"Market analyzer not available: {str(market_error)}")
            
            # Fallback to database query
            query = {
                "exchange": exchange,
                "analysis_type": "market"
            }
            
            analysis = self.db.analysis_collection.find_one(query)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error getting market context: {str(e)}")
            return None
    
    def _analyze_iv_surface(self, options, current_price):
        """Analyze implied volatility surface and skew"""
        try:
            iv_data = {}
            
            for expiry in options["expiries"]:
                calls = options["calls"].get(expiry, [])
                puts = options["puts"].get(expiry, [])
                
                # Skip if no options with market data
                if not calls or not puts:
                    continue
                
                # Filter options with market data and implied volatility
                valid_calls = [c for c in calls if c.get("market_data") and "iv" in c.get("market_data", {})]
                valid_puts = [p for p in puts if p.get("market_data") and "iv" in p.get("market_data", {})]
                
                if not valid_calls or not valid_puts:
                    continue
                
                # Calculate days to expiry
                days_to_expiry = (expiry - datetime.now()).days
                
                # Find ATM options
                atm_calls = sorted(valid_calls, key=lambda x: abs(x["strike"] - current_price))
                atm_puts = sorted(valid_puts, key=lambda x: abs(x["strike"] - current_price))
                
                if not atm_calls or not atm_puts:
                    continue
                
                atm_call = atm_calls[0]
                atm_put = atm_puts[0]
                
                # Get ATM IV
                atm_call_iv = atm_call["market_data"].get("iv")
                atm_put_iv = atm_put["market_data"].get("iv")
                
                # Average the two for ATM IV
                atm_iv = (atm_call_iv + atm_put_iv) / 2 if atm_call_iv and atm_put_iv else (atm_call_iv or atm_put_iv)
                
                # Calculate IV skew (25-delta wings)
                otm_calls = [c for c in valid_calls if c["strike"] > current_price * 1.05]
                otm_puts = [p for p in valid_puts if p["strike"] < current_price * 0.95]
                
                # Find approx 25-delta options
                call_wing = None
                put_wing = None
                
                for call in otm_calls:
                    delta = abs(call["market_data"].get("delta", 0) or 0)
                    if 0.2 <= delta <= 0.3:
                        call_wing = call
                        break
                
                for put in otm_puts:
                    delta = abs(put["market_data"].get("delta", 0) or 0)
                    if 0.2 <= delta <= 0.3:
                        put_wing = put
                        break
                
                # If no options found in delta range, use closest OTM options
                if not call_wing and otm_calls:
                    call_wing = otm_calls[0]
                if not put_wing and otm_puts:
                    put_wing = otm_puts[0]
                
                # Calculate skew measures
                skew = None
                call_skew = None
                put_skew = None
                
                if call_wing and "iv" in call_wing["market_data"]:
                    call_skew = call_wing["market_data"]["iv"] - atm_iv
                
                if put_wing and "iv" in put_wing["market_data"]:
                    put_skew = put_wing["market_data"]["iv"] - atm_iv
                
                if call_skew is not None and put_skew is not None:
                    skew = put_skew - call_skew
                
                # Store data for this expiry
                iv_data[expiry] = {
                    "days_to_expiry": days_to_expiry,
                    "atm_iv": atm_iv,
                    "call_skew": call_skew,
                    "put_skew": put_skew,
                    "skew": skew
                }
            
            # Find front-month expiry
            front_month = min(iv_data.keys()) if iv_data else None
            
            # Calculate IV percentile
            iv_percentile = self._calculate_iv_percentile(options["underlying"], options["exchange"], atm_iv)
            
            return {
                "surface": iv_data,
                "front_month": front_month,
                "atm_iv": iv_data.get(front_month, {}).get("atm_iv") if front_month else None,
                "skew": iv_data.get(front_month, {}).get("skew") if front_month else None,
                "iv_percentile": iv_percentile
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing IV surface: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_iv_percentile(self, symbol, exchange, current_iv):
        """Calculate IV percentile from historical data"""
        try:
            if not current_iv:
                return None
                
            # Query for historical IV data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "data_type": "implied_volatility"
            }
            
            # Get the last 252 days (1 year) of data
            iv_history = list(self.db.market_metadata_collection.find(
                query,
                {"date": 1, "value": 1}
            ).sort("date", -1).limit(252))
            
            if not iv_history:
                return None
            
            # Extract values
            iv_values = [h["value"] for h in iv_history]
            
            # Calculate percentile
            count_below = sum(1 for iv in iv_values if iv < current_iv)
            percentile = (count_below / len(iv_values)) * 100
            
            return percentile
            
        except Exception as e:
            self.logger.error(f"Error calculating IV percentile: {str(e)}")
            return None
    
    def _analyze_option_metrics(self, options, current_price):
        """Analyze option greeks and other metrics"""
        try:
            metrics = {}
            
            for expiry in options["expiries"]:
                calls = options["calls"].get(expiry, [])
                puts = options["puts"].get(expiry, [])
                
                # Skip if no options with market data
                if not calls or not puts:
                    continue
                
                # Filter options with market data
                valid_calls = [c for c in calls if c.get("market_data")]
                valid_puts = [p for p in puts if p.get("market_data")]
                
                if not valid_calls or not valid_puts:
                    continue
                
                # Calculate days to expiry
                days_to_expiry = (expiry - datetime.now()).days
                
                # Find ATM options
                atm_calls = sorted(valid_calls, key=lambda x: abs(x["strike"] - current_price))
                atm_puts = sorted(valid_puts, key=lambda x: abs(x["strike"] - current_price))
                
                if not atm_calls or not atm_puts:
                    continue
                
                atm_call = atm_calls[0]
                atm_put = atm_puts[0]
                
                # Get ATM greeks
                atm_call_delta = atm_call["market_data"].get("delta")
                atm_call_gamma = atm_call["market_data"].get("gamma")
                atm_call_theta = atm_call["market_data"].get("theta")
                atm_call_vega = atm_call["market_data"].get("vega")
                
                atm_put_delta = atm_put["market_data"].get("delta")
                atm_put_gamma = atm_put["market_data"].get("gamma")
                atm_put_theta = atm_put["market_data"].get("theta")
                atm_put_vega = atm_put["market_data"].get("vega")
                
                # Find options with highest metrics
                gamma_call = max(valid_calls, key=lambda x: x["market_data"].get("gamma", 0) or 0)
                gamma_put = max(valid_puts, key=lambda x: x["market_data"].get("gamma", 0) or 0)
                
                theta_call = min(valid_calls, key=lambda x: x["market_data"].get("theta", 0) or 0)
                theta_put = min(valid_puts, key=lambda x: x["market_data"].get("theta", 0) or 0)
                
                vega_call = max(valid_calls, key=lambda x: x["market_data"].get("vega", 0) or 0)
                vega_put = max(valid_puts, key=lambda x: x["market_data"].get("vega", 0) or 0)
                
                # Calculate put-call parity
                atm_call_price = atm_call["market_data"].get("close")
                atm_put_price = atm_put["market_data"].get("close")
                
                parity_diff = None
                if atm_call_price is not None and atm_put_price is not None:
                    # Simplified put-call parity: C - P = S - K*e^(-rt)
                    strike = atm_call["strike"]
                    r = 0.05  # Assumed risk-free rate
                    t = days_to_expiry / 365
                    
                    theoretical_diff = current_price - strike * math.exp(-r * t)
                    actual_diff = atm_call_price - atm_put_price
                    
                    parity_diff = actual_diff - theoretical_diff
                
                # Store metrics for this expiry
                metrics[expiry] = {
                    "days_to_expiry": days_to_expiry,
                    "atm_call": {
                        "strike": atm_call["strike"],
                        "price": atm_call_price,
                        "delta": atm_call_delta,
                        "gamma": atm_call_gamma,
                        "theta": atm_call_theta,
                        "vega": atm_call_vega
                    },
                    "atm_put": {
                        "strike": atm_put["strike"],
                        "price": atm_put_price,
                        "delta": atm_put_delta,
                        "gamma": atm_put_gamma,
                        "theta": atm_put_theta,
                        "vega": atm_put_vega
                    },
                    "highest_gamma": {
                        "call": {
                            "strike": gamma_call["strike"],
                            "gamma": gamma_call["market_data"].get("gamma")
                        },
                        "put": {
                            "strike": gamma_put["strike"],
                            "gamma": gamma_put["market_data"].get("gamma")
                        }
                    },
                    "highest_theta": {
                        "call": {
                            "strike": theta_call["strike"],
                            "theta": theta_call["market_data"].get("theta")
                        },
                        "put": {
                            "strike": theta_put["strike"],
                            "theta": theta_put["market_data"].get("theta")
                        }
                    },
                    "highest_vega": {
                        "call": {
                            "strike": vega_call["strike"],
                            "vega": vega_call["market_data"].get("vega")
                        },
                        "put": {
                            "strike": vega_put["strike"],
                            "vega": vega_put["market_data"].get("vega")
                        }
                    },
                    "put_call_parity_diff": parity_diff
                }
            
            # Find front-month expiry
            front_month = min(metrics.keys()) if metrics else None
            
            return {
                "metrics_by_expiry": metrics,
                "front_month": front_month
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing option metrics: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_volume_oi(self, options):
        """Analyze option volume and open interest patterns"""
        try:
            volume_oi = {}
            
            for expiry in options["expiries"]:
                calls = options["calls"].get(expiry, [])
                puts = options["puts"].get(expiry, [])
                
                # Skip if no options with market data
                if not calls or not puts:
                    continue
                
                # Filter options with market data
                valid_calls = [c for c in calls if c.get("market_data") and "volume" in c.get("market_data", {})]
                valid_puts = [p for p in puts if p.get("market_data") and "volume" in p.get("market_data", {})]
                
                if not valid_calls or not valid_puts:
                    continue
                
                # Calculate total volume and OI
                call_volume = sum(c["market_data"].get("volume", 0) or 0 for c in valid_calls)
                put_volume = sum(p["market_data"].get("volume", 0) or 0 for p in valid_puts)
                
                call_oi = sum(c["market_data"].get("open_interest", 0) or 0 for c in valid_calls)
                put_oi = sum(p["market_data"].get("open_interest", 0) or 0 for p in valid_puts)
                
                # Calculate PCR
                volume_pcr = put_volume / call_volume if call_volume > 0 else None
                oi_pcr = put_oi / call_oi if call_oi > 0 else None
                
                # Find strikes with unusual activity
                unusual_activity = []
                
                for option in valid_calls + valid_puts:
                    volume = option["market_data"].get("volume", 0) or 0
                    oi = option["market_data"].get("open_interest", 0) or 0
                    
                    if volume > 0 and oi > 0:
                        vol_oi_ratio = volume / oi
                        
                        if vol_oi_ratio > 0.5:  # Threshold for unusual activity
                            unusual_activity.append({
                                "strike": option["strike"],
                                "option_type": option["option_type"],
                                "volume": volume,
                                "open_interest": oi,
                                "volume_oi_ratio": vol_oi_ratio
                            })
                
                # Find options with highest volume and OI
                if valid_calls:
                    highest_vol_call = max(valid_calls, key=lambda x: x["market_data"].get("volume", 0) or 0)
                    highest_oi_call = max(valid_calls, key=lambda x: x["market_data"].get("open_interest", 0) or 0)
                else:
                    highest_vol_call = None
                    highest_oi_call = None
                
                if valid_puts:
                    highest_vol_put = max(valid_puts, key=lambda x: x["market_data"].get("volume", 0) or 0)
                    highest_oi_put = max(valid_puts, key=lambda x: x["market_data"].get("open_interest", 0) or 0)
                else:
                    highest_vol_put = None
                    highest_oi_put = None
                
                # Store volume/OI data for this expiry
                volume_oi[expiry] = {
                    "call_volume": call_volume,
                    "put_volume": put_volume,
                    "call_oi": call_oi,
                    "put_oi": put_oi,
                    "volume_pcr": volume_pcr,
                    "oi_pcr": oi_pcr,
                    "unusual_activity": sorted(unusual_activity, key=lambda x: x["volume_oi_ratio"], reverse=True)[:5],
                    "highest_volume": {
                        "call": {
                            "strike": highest_vol_call["strike"] if highest_vol_call else None,
                            "volume": highest_vol_call["market_data"].get("volume") if highest_vol_call else None
                        },
                        "put": {
                            "strike": highest_vol_put["strike"] if highest_vol_put else None,
                            "volume": highest_vol_put["market_data"].get("volume") if highest_vol_put else None
                        }
                    },
                    "highest_oi": {
                        "call": {
                            "strike": highest_oi_call["strike"] if highest_oi_call else None,
                            "oi": highest_oi_call["market_data"].get("open_interest") if highest_oi_call else None
                        },
                        "put": {
                            "strike": highest_oi_put["strike"] if highest_oi_put else None,
                            "oi": highest_oi_put["market_data"].get("open_interest") if highest_oi_put else None
                        }
                    }
                }
            
            # Find front-month expiry
            front_month = min(volume_oi.keys()) if volume_oi else None
            
            # Calculate PCR percentiles
            pcr_percentile = None
            if front_month and "oi_pcr" in volume_oi[front_month]:
                pcr_percentile = self._calculate_pcr_percentile(
                    options["underlying"], 
                    options["exchange"],
                    volume_oi[front_month]["oi_pcr"]
                )
            
            return {
                "volume_oi_by_expiry": volume_oi,
                "front_month": front_month,
                "front_month_oi_pcr": volume_oi.get(front_month, {}).get("oi_pcr") if front_month else None,
                "pcr_percentile": pcr_percentile
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume and OI: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_pcr_percentile(self, symbol, exchange, current_pcr):
        """Calculate PCR percentile from historical data"""
        try:
            if not current_pcr:
                return None
                
            # Query for historical PCR data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "data_type": "put_call_ratio"
            }
            
            # Get the last 60 days of data
            pcr_history = list(self.db.market_metadata_collection.find(
                query,
                {"date": 1, "value": 1}
            ).sort("date", -1).limit(60))
            
            if not pcr_history:
                return None
            
            # Extract values
            pcr_values = [h["value"] for h in pcr_history]
            
            # Calculate percentile
            count_below = sum(1 for pcr in pcr_values if pcr < current_pcr)
            percentile = (count_below / len(pcr_values)) * 100
            
            return percentile
            
        except Exception as e:
            self.logger.error(f"Error calculating PCR percentile: {str(e)}")
            return None
    
    def _determine_best_strategy(self, prediction, technical, iv_data, 
                               market_context, option_metrics, current_price, expiry):
        """Determine the best option strategy based on market conditions"""
        try:
            # Default strategy if nothing else matches
            strategy = "long_call"
            
            # Get key metrics
            iv_percentile = iv_data.get("iv_percentile")
            skew = iv_data.get("skew")
            days_to_expiry = (expiry - datetime.now()).days
            
            # Get directional bias from prediction or technical analysis
            direction = None
            
            if prediction and "prediction" in prediction:
                direction = "bullish" if prediction["prediction"] == "up" else \
                           "bearish" if prediction["prediction"] == "down" else "neutral"
            elif technical and "signals" in technical:
                # Count bullish vs bearish signals
                bullish_count = sum(1 for s in technical["signals"] if s.get("direction") == "bullish")
                bearish_count = sum(1 for s in technical["signals"] if s.get("direction") == "bearish")
                
                if bullish_count > bearish_count + 1:
                    direction = "bullish"
                elif bearish_count > bullish_count + 1:
                    direction = "bearish"
                else:
                    direction = "neutral"
            else:
                # Default to neutral if no directional info available
                direction = "neutral"
            
            # Check for volatility expectations
            vol_expectation = "steady"
            
            if iv_percentile is not None:
                if iv_percentile > 80:
                    vol_expectation = "decrease"
                elif iv_percentile < 20:
                    vol_expectation = "increase"
            
            # Determine best strategy based on outlook
            if direction == "bullish":
                if vol_expectation == "increase":
                    if days_to_expiry >= 30:
                        strategy = "call_backspread"  # Bullish and long vega
                    else:
                        strategy = "long_call"  # Simpler for shorter expiries
                elif vol_expectation == "decrease":
                    strategy = "bull_call_spread"  # Bullish and short vega
                else:
                    strategy = "long_call"  # Basic bullish strategy
            
            elif direction == "bearish":
                if vol_expectation == "increase":
                    if days_to_expiry >= 30:
                        strategy = "put_backspread"  # Bearish and long vega
                    else:
                        strategy = "long_put"  # Simpler for shorter expiries
                elif vol_expectation == "decrease":
                    strategy = "bear_put_spread"  # Bearish and short vega
                else:
                    strategy = "long_put"  # Basic bearish strategy
            
            else:  # Neutral
                if vol_expectation == "increase":
                    strategy = "long_straddle"  # Neutral with long vega
                elif vol_expectation == "decrease":
                    if days_to_expiry >= 21:
                        strategy = "iron_condor"  # Neutral with short vega
                    else:
                        strategy = "short_straddle"  # Short vega but higher risk
                else:
                    if abs(skew or 0) > 0.05:  # Significant skew
                        strategy = "ratio_spread"  # Take advantage of skew
                    else:
                        strategy = "covered_call"  # Conservative neutral strategy
            
            # Special case for very close to expiry
            if days_to_expiry <= 5:
                if direction == "bullish":
                    strategy = "long_call"  # Simplest for short term bullish
                elif direction == "bearish":
                    strategy = "long_put"  # Simplest for short term bearish
                else:
                    strategy = "long_straddle"  # Simplest for short term volatility play
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error determining best strategy: {str(e)}")
            return "long_call"  # Default to simplest strategy on error
    
    def _get_strategy_parameters(self, strategy, symbol, exchange, options, 
                               current_price, expiry, technical, iv_data, 
                               market_context, option_metrics):
        """Get specific parameters for the selected option strategy"""
        try:
            # Default parameters
            params = {
                "suitable_for_trading": False,
                "reason": "Unable to find suitable options for strategy",
                "strategy": strategy
            }
            
            # Get available options for this expiry
            calls = options["calls"].get(expiry, [])
            puts = options["puts"].get(expiry, [])
            
            # Filter options with market data
            valid_calls = [c for c in calls if c.get("market_data") and "close" in c.get("market_data", {})]
            valid_puts = [p for p in puts if p.get("market_data") and "close" in p.get("market_data", {})]
            
            if not valid_calls or not valid_puts:
                return params
            
            # Sort options by strike
            valid_calls = sorted(valid_calls, key=lambda x: x["strike"])
            valid_puts = sorted(valid_puts, key=lambda x: x["strike"])
            
            # Find ATM and near strikes
            atm_index_calls = min(range(len(valid_calls)), key=lambda i: abs(valid_calls[i]["strike"] - current_price))
            atm_index_puts = min(range(len(valid_puts)), key=lambda i: abs(valid_puts[i]["strike"] - current_price))
            
            atm_call = valid_calls[atm_index_calls]
            atm_put = valid_puts[atm_index_puts]
            
            # Days to expiry
            days_to_expiry = (expiry - datetime.now()).days
            
            # Strategy specific logic
            if strategy == "long_call":
                # Find slightly OTM call with good liquidity
                otm_calls = [c for c in valid_calls if c["strike"] > current_price]
                if not otm_calls:
                    return params
                
                # Sort by volume to find liquid options
                liquid_calls = sorted(otm_calls, key=lambda x: x["market_data"].get("volume", 0) or 0, reverse=True)
                
                # Take the first OTM call with decent volume
                target_call = liquid_calls[0] if liquid_calls and liquid_calls[0]["market_data"].get("volume", 0) > 10 else otm_calls[0]
                
                # Calculate risk metrics
                entry_price = target_call["market_data"]["close"]
                max_loss = entry_price * 100  # Per contract
                break_even = target_call["strike"] + entry_price
                
                # Calculate probability of profit using delta as approximation
                delta = target_call["market_data"].get("delta", 0.3) or 0.3
                profit_probability = delta * 100
                
                params = {
                    "suitable_for_trading": True,
                    "strategy": strategy,
                    "leg1": {
                        "option_type": "call",
                        "strike": target_call["strike"],
                        "action": "buy",
                        "price": entry_price,
                        "delta": delta,
                        "gamma": target_call["market_data"].get("gamma"),
                        "theta": target_call["market_data"].get("theta"),
                        "vega": target_call["market_data"].get("vega"),
                        "volume": target_call["market_data"].get("volume"),
                        "open_interest": target_call["market_data"].get("open_interest")
                    },
                    "max_profit": "Unlimited",
                    "max_loss": max_loss,
                    "break_even": break_even,
                    "profit_probability": profit_probability,
                    "risk_reward": "Unlimited",
                    "days_to_expiry": days_to_expiry
                }
            
            elif strategy == "long_put":
                # Find slightly OTM put with good liquidity
                otm_puts = [p for p in valid_puts if p["strike"] < current_price]
                if not otm_puts:
                    return params
                
                # Sort by volume to find liquid options
                liquid_puts = sorted(otm_puts, key=lambda x: x["market_data"].get("volume", 0) or 0, reverse=True)
                
                # Take the first OTM put with decent volume
                target_put = liquid_puts[0] if liquid_puts and liquid_puts[0]["market_data"].get("volume", 0) > 10 else otm_puts[0]
                
                # Calculate risk metrics
                entry_price = target_put["market_data"]["close"]
                max_loss = entry_price * 100  # Per contract
                break_even = target_put["strike"] - entry_price
                
                # Calculate probability of profit using delta as approximation
                delta = abs(target_put["market_data"].get("delta", 0.3) or 0.3)
                profit_probability = delta * 100
                
                params = {
                    "suitable_for_trading": True,
                    "strategy": strategy,
                    "leg1": {
                        "option_type": "put",
                        "strike": target_put["strike"],
                        "action": "buy",
                        "price": entry_price,
                        "delta": -delta,
                        "gamma": target_put["market_data"].get("gamma"),
                        "theta": target_put["market_data"].get("theta"),
                        "vega": target_put["market_data"].get("vega"),
                        "volume": target_put["market_data"].get("volume"),
                        "open_interest": target_put["market_data"].get("open_interest")
                    },
                    "max_profit": target_put["strike"] * 100,  # Maximum if stock goes to zero
                    "max_loss": max_loss,
                    "break_even": break_even,
                    "profit_probability": profit_probability,
                    "risk_reward": (target_put["strike"] / entry_price) - 1,
                    "days_to_expiry": days_to_expiry
                }
            
            elif strategy == "bull_call_spread":
                # Find OTM calls for spread
                otm_calls = [c for c in valid_calls if c["strike"] >= current_price]
                if len(otm_calls) < 2:
                    return params
                
                # Long the first OTM call
                long_call = otm_calls[0]
                
                # Short a further OTM call (target 5-10% OTM from current)
                target_pct = 0.075  # 7.5% OTM
                target_strike = current_price * (1 + target_pct)
                
                short_call_candidates = [c for c in otm_calls if c["strike"] > long_call["strike"]]
                if not short_call_candidates:
                    return params
                
                short_call = min(short_call_candidates, key=lambda c: abs(c["strike"] - target_strike))
                
                # Calculate risk metrics
                long_price = long_call["market_data"]["close"]
                short_price = short_call["market_data"]["close"]
                
                net_debit = long_price - short_price
                max_profit = (short_call["strike"] - long_call["strike"]) - net_debit
                max_loss = net_debit * 100  # Per contract
                break_even = long_call["strike"] + net_debit
                
                # Calculate probability of profit
                long_delta = long_call["market_data"].get("delta", 0.5) or 0.5
                short_delta = short_call["market_data"].get("delta", 0.3) or 0.3
                profit_probability = (long_delta - short_delta) * 100
                
                params = {
                    "suitable_for_trading": True,
                    "strategy": strategy,
                    "leg1": {
                        "option_type": "call",
                        "strike": long_call["strike"],
                        "action": "buy",
                        "price": long_price,
                        "delta": long_call["market_data"].get("delta"),
                        "gamma": long_call["market_data"].get("gamma"),
                        "theta": long_call["market_data"].get("theta"),
                        "vega": long_call["market_data"].get("vega")
                    },
                    "leg2": {
                        "option_type": "call",
                        "strike": short_call["strike"],
                        "action": "sell",
                        "price": short_price,
                        "delta": short_call["market_data"].get("delta"),
                        "gamma": short_call["market_data"].get("gamma"),
                        "theta": short_call["market_data"].get("theta"),
                        "vega": short_call["market_data"].get("vega")
                    },
                    "max_profit": max_profit * 100,  # Per contract
                    "max_loss": max_loss,
                    "net_debit": net_debit * 100,
                    "break_even": break_even,
                    "profit_probability": profit_probability,
                    "risk_reward": max_profit / net_debit,
                    "days_to_expiry": days_to_expiry
                }
            
            elif strategy == "bear_put_spread":
                # Find OTM puts for spread
                otm_puts = [p for p in valid_puts if p["strike"] <= current_price]
                if len(otm_puts) < 2:
                    return params
                
                # Reverse sort for puts (highest strike first)
                otm_puts = sorted(otm_puts, key=lambda p: p["strike"], reverse=True)
                
                # Long the first OTM put
                long_put = otm_puts[0]
                
                # Short a further OTM put (target 5-10% OTM from current)
                target_pct = 0.075  # 7.5% OTM
                target_strike = current_price * (1 - target_pct)
                
                short_put_candidates = [p for p in otm_puts if p["strike"] < long_put["strike"]]
                if not short_put_candidates:
                    return params
                
                short_put = min(short_put_candidates, key=lambda p: abs(p["strike"] - target_strike))
                
                # Calculate risk metrics
                long_price = long_put["market_data"]["close"]
                short_price = short_put["market_data"]["close"]
                
                net_debit = long_price - short_price
                max_profit = (long_put["strike"] - short_put["strike"]) - net_debit
                max_loss = net_debit * 100  # Per contract
                break_even = long_put["strike"] - net_debit
                
                # Calculate probability of profit
                long_delta = abs(long_put["market_data"].get("delta", 0.5) or 0.5)
                short_delta = abs(short_put["market_data"].get("delta", 0.3) or 0.3)
                profit_probability = (long_delta - short_delta) * 100
                
                params = {
                    "suitable_for_trading": True,
                    "strategy": strategy,
                    "leg1": {
                        "option_type": "put",
                        "strike": long_put["strike"],
                        "action": "buy",
                        "price": long_price,
                        "delta": long_put["market_data"].get("delta"),
                        "gamma": long_put["market_data"].get("gamma"),
                        "theta": long_put["market_data"].get("theta"),
                        "vega": long_put["market_data"].get("vega")
                    },
                    "leg2": {
                        "option_type": "put",
                        "strike": short_put["strike"],
                        "action": "sell",
                        "price": short_price,
                        "delta": short_put["market_data"].get("delta"),
                        "gamma": short_put["market_data"].get("gamma"),
                        "theta": short_put["market_data"].get("theta"),
                        "vega": short_put["market_data"].get("vega")
                    },
                    "max_profit": max_profit * 100,  # Per contract
                    "max_loss": max_loss,
                    "net_debit": net_debit * 100,
                    "break_even": break_even,
                    "profit_probability": profit_probability,
                    "risk_reward": max_profit / net_debit,
                    "days_to_expiry": days_to_expiry
                }
            
            elif strategy == "long_straddle":
                # Use ATM options for straddle
                
                # Calculate risk metrics
                call_price = atm_call["market_data"]["close"]
                put_price = atm_put["market_data"]["close"]
                
                net_debit = call_price + put_price
                max_loss = net_debit * 100  # Per contract
                
                # Calculate break-even points
                upper_break_even = atm_call["strike"] + net_debit
                lower_break_even = atm_put["strike"] - net_debit
                
                # Estimate probability based on expected move
                # Using rough estimate that stock will move more than net_debit with 50% probability
                profit_probability = 50
                
                params = {
                    "suitable_for_trading": True,
                    "strategy": strategy,
                    "leg1": {
                        "option_type": "call",
                        "strike": atm_call["strike"],
                        "action": "buy",
                        "price": call_price,
                        "delta": atm_call["market_data"].get("delta"),
                        "gamma": atm_call["market_data"].get("gamma"),
                        "theta": atm_call["market_data"].get("theta"),
                        "vega": atm_call["market_data"].get("vega")
                    },
                    "leg2": {
                        "option_type": "put",
                        "strike": atm_put["strike"],
                        "action": "buy",
                        "price": put_price,
                        "delta": atm_put["market_data"].get("delta"),
                        "gamma": atm_put["market_data"].get("gamma"),
                        "theta": atm_put["market_data"].get("theta"),
                        "vega": atm_put["market_data"].get("vega")
                    },
                    "max_profit": "Unlimited",
                    "max_loss": max_loss,
                    "net_debit": net_debit * 100,
                    "upper_break_even": upper_break_even,
                    "lower_break_even": lower_break_even,
                    "profit_probability": profit_probability,
                    "risk_reward": "Unlimited",
                    "days_to_expiry": days_to_expiry
                }
            
            elif strategy == "iron_condor":
                # Need sufficient strikes for 4-legged strategy
                if len(valid_calls) < 3 or len(valid_puts) < 3:
                    return params
                
                # Target 30-delta wings for standard iron condor
                target_delta = 0.30
                
                # Find call wing
                otm_calls = [c for c in valid_calls if c["strike"] > current_price]
                if len(otm_calls) < 2:
                    return params
                
                # Sort calls by delta (closest to target first)
                otm_calls_with_delta = [c for c in otm_calls if "delta" in c["market_data"]]
                if not otm_calls_with_delta:
                    # Use distance from ATM as proxy if no delta available
                    short_call = otm_calls[0]  # First OTM
                    long_call = otm_calls[1]   # Second OTM
                else:
                    short_call = min(otm_calls_with_delta, key=lambda c: abs(c["market_data"]["delta"] - target_delta))
                    # Long call should be higher strike
                    long_call_candidates = [c for c in otm_calls if c["strike"] > short_call["strike"]]
                    if not long_call_candidates:
                        return params
                    long_call = long_call_candidates[0]
                
                # Find put wing
                otm_puts = [p for p in valid_puts if p["strike"] < current_price]
                if len(otm_puts) < 2:
                    return params
                
                # Sort puts by delta (closest to target first)
                otm_puts_with_delta = [p for p in otm_puts if "delta" in p["market_data"]]
                if not otm_puts_with_delta:
                    # Use distance from ATM as proxy if no delta available
                    short_put = otm_puts[0]   # First OTM
                    long_put = otm_puts[1]    # Second OTM
                else:
                    short_put = min(otm_puts_with_delta, key=lambda p: abs(abs(p["market_data"]["delta"]) - target_delta))
                    # Long put should be lower strike
                    long_put_candidates = [p for p in otm_puts if p["strike"] < short_put["strike"]]
                    if not long_put_candidates:
                        return params
                    long_put = long_put_candidates[0]
                
                # Calculate risk metrics
                short_call_price = short_call["market_data"]["close"]
                long_call_price = long_call["market_data"]["close"]
                short_put_price = short_put["market_data"]["close"]
                long_put_price = long_put["market_data"]["close"]
                
                # Net credit received
                net_credit = (short_call_price - long_call_price) + (short_put_price - long_put_price)
                
                # Calculate max profit and loss
                call_wing_width = long_call["strike"] - short_call["strike"]
                put_wing_width = short_put["strike"] - long_put["strike"]
                
                max_profit = net_credit * 100  # Per contract
                max_loss = (min(call_wing_width, put_wing_width) - net_credit) * 100
                
                # Calculate break-even points
                upper_break_even = short_call["strike"] + net_credit
                lower_break_even = short_put["strike"] - net_credit
                
                # Calculate probability of profit (rough estimate)
                short_call_delta = abs(short_call["market_data"].get("delta", target_delta) or target_delta)
                short_put_delta = abs(short_put["market_data"].get("delta", target_delta) or target_delta)
                
                profit_probability = (1 - short_call_delta - short_put_delta) * 100
                
                params = {
                    "suitable_for_trading": True,
                    "strategy": strategy,
                    "leg1": {
                        "option_type": "put",
                        "strike": long_put["strike"],
                        "action": "buy",
                        "price": long_put_price
                    },
                    "leg2": {
                        "option_type": "put",
                        "strike": short_put["strike"],
                        "action": "sell",
                        "price": short_put_price
                    },
                    "leg3": {
                        "option_type": "call",
                        "strike": short_call["strike"],
                        "action": "sell",
                        "price": short_call_price
                    },
                    "leg4": {
                        "option_type": "call",
                        "strike": long_call["strike"],
                        "action": "buy",
                        "price": long_call_price
                    },
                    "max_profit": max_profit,
                    "max_loss": max_loss,
                    "net_credit": net_credit * 100,
                    "upper_break_even": upper_break_even,
                    "lower_break_even": lower_break_even,
                    "profit_probability": profit_probability,
                    "risk_reward": net_credit / (min(call_wing_width, put_wing_width) - net_credit),
                    "days_to_expiry": days_to_expiry
                }
            
            # Add other strategies as needed
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error getting strategy parameters: {str(e)}")
            return {
                "suitable_for_trading": False,
                "reason": f"Error: {str(e)}",
                "strategy": strategy
            }
    
    def _save_opportunity(self, opportunity):
        """Save trading opportunity to database"""
        try:
            # Create document for database
            doc = {
                "symbol": opportunity["symbol"],
                "exchange": opportunity["exchange"],
                "opportunity_type": "options",
                "expiry": opportunity["expiry"],
                "timestamp": opportunity["timestamp"],
                "data": opportunity
            }
            
            # Insert into database
            self.db.opportunity_collection.insert_one(doc)
            
        except Exception as e:
            self.logger.error(f"Error saving opportunity: {str(e)}")
    
    def execute_trade(self, symbol, exchange, strategy, strategy_params, order_params=None):
        """
        Execute an options trade based on identified opportunity
        
        Parameters:
        - symbol: Underlying symbol
        - exchange: Exchange (e.g., NSE)
        - strategy: Option strategy to execute
        - strategy_params: Parameters for the strategy
        - order_params: Additional order parameters
        
        Returns:
        - Dictionary with order execution details
        """
        try:
            # Default to simulation mode
            if not order_params:
                order_params = {}
            
            simulation = order_params.get("simulation", True)
            
            # Extract key parameters
            expiry = strategy_params.get("days_to_expiry")
            legs = []
            
            # Extract legs
            for i in range(1, 5):
                leg_key = f"leg{i}"
                if leg_key in strategy_params:
                    legs.append(strategy_params[leg_key])
            
            if not legs:
                return {"status": "error", "message": "No strategy legs found"}
            
            if simulation:
                # Simulated trade execution
                trade_result = {
                    "status": "success",
                    "order_id": f"sim_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "symbol": symbol,
                    "exchange": exchange,
                    "strategy": strategy,
                    "timestamp": datetime.now(),
                    "legs": legs,
                    "max_profit": strategy_params.get("max_profit"),
                    "max_loss": strategy_params.get("max_loss"),
                    "net_debit": strategy_params.get("net_debit"),
                    "net_credit": strategy_params.get("net_credit"),
                    "break_even": strategy_params.get("break_even"),
                    "upper_break_even": strategy_params.get("upper_break_even"),
                    "lower_break_even": strategy_params.get("lower_break_even"),
                    "simulated": True
                }
                
                # Log the simulated trade
                self.logger.info(f"Simulated options trade: {trade_result}")
            else:
                # Real trading via broker integration
                try:
                    from realtime.zerodha_integration import ZerodhaTrader
                    trader = ZerodhaTrader(self.db)
                    
                    # Execute each leg
                    leg_results = []
                    
                    for leg in legs:
                        option_type = leg["option_type"]
                        strike = leg["strike"]
                        action = leg["action"]
                        
                        transaction_type = "BUY" if action == "buy" else "SELL"
                        
                        # Place the order
                        result = trader.place_option_order(
                            symbol=symbol,
                            exchange=exchange,
                            expiry=expiry,
                            option_type=option_type,
                            strike=strike,
                            transaction_type=transaction_type,
                            order_type="MARKET",
                            quantity=1  # Default to 1 lot
                        )
                        
                        if result["status"] == "success":
                            leg["order_id"] = result["order_id"]
                            leg["executed_price"] = result.get("executed_price")
                            leg_results.append(result)
                        else:
                            # Handle failure
                            # Close any open positions from previous legs
                            for executed_leg in leg_results:
                                trader.place_option_order(
                                    symbol=symbol,
                                    exchange=exchange,
                                    expiry=expiry,
                                    option_type=executed_leg["option_type"],
                                    strike=executed_leg["strike"],
                                    transaction_type="SELL" if executed_leg["transaction_type"] == "BUY" else "BUY",
                                    order_type="MARKET",
                                    quantity=1
                                )
                            
                            return {"status": "error", "message": f"Failed to execute leg: {result.get('message')}"}
                    
                    # If all legs executed successfully
                    trade_result = {
                        "status": "success",
                        "order_id": f"multi_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "symbol": symbol,
                        "exchange": exchange,
                        "strategy": strategy,
                        "timestamp": datetime.now(),
                        "legs": legs,
                        "max_profit": strategy_params.get("max_profit"),
                        "max_loss": strategy_params.get("max_loss"),
                        "net_debit": strategy_params.get("net_debit"),
                        "net_credit": strategy_params.get("net_credit"),
                        "break_even": strategy_params.get("break_even"),
                        "upper_break_even": strategy_params.get("upper_break_even"),
                        "lower_break_even": strategy_params.get("lower_break_even"),
                        "simulated": False
                    }
                except Exception as broker_error:
                    self.logger.error(f"Error executing trade via broker: {str(broker_error)}")
                    return {"status": "error", "message": f"Broker execution error: {str(broker_error)}"}
            
            # Save trade to database
            self._save_trade(trade_result)
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Error executing options trade: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _save_trade(self, trade_result):
        """Save trade execution to database"""
        try:
            # Create document for database
            doc = {
                "symbol": trade_result["symbol"],
                "exchange": trade_result["exchange"],
                "trade_type": "options",
                "strategy": trade_result["strategy"],
                "entry_time": trade_result["timestamp"],
                "legs": trade_result["legs"],
                "max_profit": trade_result.get("max_profit"),
                "max_loss": trade_result.get("max_loss"),
                "net_debit": trade_result.get("net_debit"),
                "net_credit": trade_result.get("net_credit"),
                "break_even": trade_result.get("break_even"),
                "upper_break_even": trade_result.get("upper_break_even"),
                "lower_break_even": trade_result.get("lower_break_even"),
                "order_id": trade_result["order_id"],
                "status": "open",
                "simulated": trade_result.get("simulated", True)
            }
            
            # Insert into database
            self.db.trade_collection.insert_one(doc)
            
        except Exception as e:
            self.logger.error(f"Error saving trade: {str(e)}")
    
    def close_options_position(self, trade_id, reason="manual", simulation=True):
        """
        Close an existing options trade
        
        Parameters:
        - trade_id: Trade ID to close
        - reason: Reason for closing the trade
        - simulation: Whether this is a simulated trade
        
        Returns:
        - Dictionary with closure result
        """
        try:
            # Get the trade from database
            trade = self.db.trade_collection.find_one({"_id": trade_id})
            
            if not trade:
                return {"status": "error", "message": "Trade not found"}
            
            if trade["status"] != "open":
                return {"status": "error", "message": "Trade already closed"}
            
            if simulation or trade.get("simulated", True):
                # Simulated closure
                
                # Get current option prices
                legs_with_prices = []
                total_exit_value = 0
                
                for leg in trade["legs"]:
                    # Get current option price
                    option_price = self._get_current_option_price(
                        trade["symbol"],
                        trade["exchange"],
                        leg.get("expiry"),
                        leg["option_type"],
                        leg["strike"]
                    )
                    
                    # If price not available, estimate based on entry price
                    if not option_price:
                        option_price = leg.get("price", 0)
                    
                    leg_exit = {
                        "option_type": leg["option_type"],
                        "strike": leg["strike"],
                        "action": "sell" if leg["action"] == "buy" else "buy",  # Reverse action
                        "entry_price": leg.get("price", 0),
                        "exit_price": option_price,
                        "profit_loss": (option_price - leg.get("price", 0)) if leg["action"] == "buy" else (leg.get("price", 0) - option_price)
                    }
                    
                    legs_with_prices.append(leg_exit)
                    
                    # Calculate contribution to total P&L
                    if leg["action"] == "buy":
                        total_exit_value += option_price
                    else:  # leg["action"] == "sell"
                        total_exit_value -= option_price
                
                # Calculate entry value
                total_entry_value = 0
                for leg in trade["legs"]:
                    if leg["action"] == "buy":
                        total_entry_value -= leg.get("price", 0)
                    else:  # leg["action"] == "sell"
                        total_entry_value += leg.get("price", 0)
                
                # Calculate total P&L
                total_profit_loss = (total_exit_value + total_entry_value) * 100  # Per contract
                
                # Update in database
                self.db.trade_collection.update_one(
                    {"_id": trade_id},
                    {
                        "$set": {
                            "status": "closed",
                            "exit_time": datetime.now(),
                            "exit_reason": reason,
                            "exit_legs": legs_with_prices,
                            "profit_loss": total_profit_loss
                        }
                    }
                )
                
                closure_result = {
                    "status": "success",
                    "trade_id": trade_id,
                    "exit_legs": legs_with_prices,
                    "profit_loss": total_profit_loss,
                    "exit_reason": reason,
                    "simulated": True
                }
                
                return closure_result
            else:
                # Real closure via broker integration
                try:
                    from realtime.zerodha_integration import ZerodhaTrader
                    trader = ZerodhaTrader(self.db)
                    
                    # Execute closure for each leg
                    legs_with_prices = []
                    
                    for leg in trade["legs"]:
                        # Reverse the action
                        transaction_type = "SELL" if leg["action"] == "buy" else "BUY"
                        
                        # Place the order
                        result = trader.place_option_order(
                            symbol=trade["symbol"],
                            exchange=trade["exchange"],
                            expiry=leg.get("expiry"),
                            option_type=leg["option_type"],
                            strike=leg["strike"],
                            transaction_type=transaction_type,
                            order_type="MARKET",
                            quantity=1  # Default to 1 lot
                        )
                        
                        if result["status"] == "success":
                            leg_exit = {
                                "option_type": leg["option_type"],
                                "strike": leg["strike"],
                                "action": "sell" if leg["action"] == "buy" else "buy",
                                "entry_price": leg.get("price", 0),
                                "exit_price": result.get("executed_price", 0),
                                "profit_loss": (result.get("executed_price", 0) - leg.get("price", 0)) if leg["action"] == "buy" else (leg.get("price", 0) - result.get("executed_price", 0)),
                                "order_id": result["order_id"]
                            }
                            
                            legs_with_prices.append(leg_exit)
                        else:
                            # Continue with partial closure if some legs fail
                            self.logger.warning(f"Failed to close leg: {result.get('message')}")
                    
                    # Calculate total P&L from closed legs
                    total_profit_loss = sum(leg["profit_loss"] for leg in legs_with_prices) * 100  # Per contract
                    
                    # Update in database
                    self.db.trade_collection.update_one(
                        {"_id": trade_id},
                        {
                            "$set": {
                                "status": "closed",
                                "exit_time": datetime.now(),
                                "exit_reason": reason,
                                "exit_legs": legs_with_prices,
                                "profit_loss": total_profit_loss,
                                "fully_closed": len(legs_with_prices) == len(trade["legs"])
                            }
                        }
                    )
                    
                    closure_result = {
                        "status": "success",
                        "trade_id": trade_id,
                        "exit_legs": legs_with_prices,
                        "profit_loss": total_profit_loss,
                        "exit_reason": reason,
                        "simulated": False,
                        "fully_closed": len(legs_with_prices) == len(trade["legs"])
                    }
                    
                    return closure_result
                    
                except Exception as broker_error:
                    self.logger.error(f"Error closing trade via broker: {str(broker_error)}")
                    return {"status": "error", "message": f"Broker closure error: {str(broker_error)}"}
            
        except Exception as e:
            self.logger.error(f"Error closing options position: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_current_option_price(self, symbol, exchange, expiry, option_type, strike):
        """Get current price for an option"""
        try:
            # Try real-time price first
            try:
                from realtime.zerodha_integration import ZerodhaTrader
                trader = ZerodhaTrader(self.db)
                price = trader.get_option_ltp(symbol, exchange, expiry, option_type, strike)
                if price:
                    return price
            except Exception as broker_error:
                self.logger.warning(f"Unable to get real-time option price: {str(broker_error)}")
            
            # Fallback to latest price from database
            query = {
                "underlying": symbol,
                "exchange": exchange,
                "expiry": expiry,
                "option_type": option_type,
                "strike": strike
            }
            
            # Find the option instrument
            option = self.db.instrument_collection.find_one(query)
            
            if not option:
                return None
            
            # Get latest market data for this option
            market_data = self.db.market_data_collection.find_one(
                {"instrument_id": option["_id"]},
                {"close": 1}
            )
            
            if market_data and "close" in market_data:
                return market_data["close"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current option price: {str(e)}")
            return None
    
    def get_options_positions(self):
        """
        Get all current options positions
        
        Returns:
        - List of open options positions with P&L estimates
        """
        try:
            # Query for open options trades
            query = {
                "trade_type": "options",
                "status": "open"
            }
            
            positions = list(self.db.trade_collection.find(query))
            
            # For each position, estimate current value and unrealized P&L
            for position in positions:
                total_current_value = 0
                
                for leg in position["legs"]:
                    # Get current option price
                    current_price = self._get_current_option_price(
                        position["symbol"],
                        position["exchange"],
                        leg.get("expiry"),
                        leg["option_type"],
                        leg["strike"]
                    )
                    
                    if current_price:
                        leg["current_price"] = current_price
                        
                        # Calculate leg P&L
                        if leg["action"] == "buy":
                            leg["unrealized_pl"] = (current_price - leg.get("price", 0)) * 100  # Per contract
                            total_current_value -= current_price
                        else:  # leg["action"] == "sell"
                            leg["unrealized_pl"] = (leg.get("price", 0) - current_price) * 100  # Per contract
                            total_current_value += current_price
                
                # Calculate total entry value
                total_entry_value = 0
                for leg in position["legs"]:
                    if leg["action"] == "buy":
                        total_entry_value += leg.get("price", 0)
                    else:  # leg["action"] == "sell"
                        total_entry_value -= leg.get("price", 0)
                
                # Calculate total P&L
                position["total_entry_value"] = total_entry_value * 100  # Per contract
                position["total_current_value"] = total_current_value * 100  # Per contract
                position["unrealized_pl"] = (total_current_value - total_entry_value) * 100  # Per contract
                
                # Calculate days to expiry for each leg
                for leg in position["legs"]:
                    if "expiry" in leg:
                        leg["days_to_expiry"] = (leg["expiry"] - datetime.now()).days
            
            return {
                "status": "success",
                "positions": positions,
                "count": len(positions)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting options positions: {str(e)}")
            return {"status": "error", "message": str(e)}