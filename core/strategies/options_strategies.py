# options_strategies.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import math

class OptionsStrategies:
    def __init__(self, db_connector):
        """Initialize the options strategies module"""
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, symbol, exchange, expiry=None):
        """
        Analyze options for potential trading opportunities
        
        Parameters:
        - symbol: The symbol to analyze
        - exchange: The exchange (e.g., NSE)
        - expiry: Optional specific expiry date to analyze
        
        Returns:
        - Dictionary with analysis results and trading signals
        """
        try:
            # Get underlying price
            underlying_price = self._get_underlying_price(symbol, exchange)
            if not underlying_price:
                return {"status": "error", "message": "Unable to get underlying price"}
            
            # Get available option chains
            option_chains = self._get_option_chains(symbol, exchange, expiry)
            if not option_chains:
                return {"status": "error", "message": "No option chains available"}
            
            # Get IV surface
            iv_surface = self._analyze_iv_surface(symbol, exchange, option_chains, underlying_price)
            
            # Analyze options metrics
            options_metrics = self._analyze_options_metrics(symbol, exchange, option_chains)
            
            # Analyze option spreads
            spread_opportunities = self._analyze_spread_opportunities(symbol, exchange, option_chains, underlying_price)
            
            # Analyze option volume and OI
            volume_oi_analysis = self._analyze_volume_oi(symbol, exchange, option_chains)
            
            # Put-call ratio analysis
            pc_ratio = self._analyze_put_call_ratio(symbol, exchange, option_chains)
            
            # Analyze skew
            skew_analysis = self._analyze_skew(symbol, exchange, option_chains, underlying_price)
            
            # Consolidate analysis
            analysis_result = {
                "symbol": symbol,
                "exchange": exchange,
                "options_analysis": {
                    "timestamp": datetime.now(),
                    "underlying_price": underlying_price,
                    "iv_surface": iv_surface,
                    "options_metrics": options_metrics,
                    "spread_opportunities": spread_opportunities,
                    "volume_oi_analysis": volume_oi_analysis,
                    "put_call_ratio": pc_ratio,
                    "skew_analysis": skew_analysis
                }
            }
            
            # Generate trading signals
            signals = self._generate_signals(analysis_result)
            analysis_result["signals"] = signals
            
            # Save analysis to database
            self._save_analysis(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing options for {symbol}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_underlying_price(self, symbol, exchange):
        """Get the latest price of the underlying"""
        try:
            # Query for the latest equity price
            latest_price = self.db.market_data_collection.find_one(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "instrument_type": "equity"
                },
                {"close": 1}
            )
            
            if latest_price:
                return latest_price["close"]
            
            # Try futures if equity not available
            latest_futures = self.db.market_data_collection.find_one(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "instrument_type": "futures"
                },
                {"close": 1}
            )
            
            if latest_futures:
                return latest_futures["close"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting underlying price: {str(e)}")
            return None
    
    def _get_option_chains(self, symbol, exchange, expiry=None):
        """Get available option chains for a symbol"""
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
            
            # Get option chain data
            options = list(self.db.instrument_collection.find(query))
            
            # If no expiry specified, group by expiry
            if not expiry:
                # Group options by expiry
                expirations = {}
                for option in options:
                    exp = option["expiry"]
                    if exp not in expirations:
                        expirations[exp] = []
                    expirations[exp].append(option)
                
                # Sort expirations by date
                sorted_expiries = sorted(expirations.keys())
                
                # Return nearest 3 expirations
                option_chains = {}
                for exp in sorted_expiries[:3]:
                    option_chains[exp] = expirations[exp]
                
                return option_chains
            
            # If expiry specified, return options for that expiry
            return {expiry: options}
            
        except Exception as e:
            self.logger.error(f"Error getting option chains: {str(e)}")
            return {}
    
    def _analyze_iv_surface(self, symbol, exchange, option_chains, underlying_price):
        """
        Analyze implied volatility surface
        """
        try:
            iv_data = {}
            
            for expiry, options in option_chains.items():
                days_to_expiry = (expiry - datetime.now()).days
                
                # Skip if less than 1 day to expiry
                if days_to_expiry < 1:
                    continue
                
                calls = []
                puts = []
                
                # Process each option
                for option in options:
                    # Get latest option data
                    option_data = self._get_option_data(option["_id"])
                    if not option_data or "iv" not in option_data:
                        continue
                    
                    strike = option["strike"]
                    iv = option_data["iv"]
                    
                    # Moneyness = strike / current price
                    moneyness = strike / underlying_price
                    
                    option_info = {
                        "strike": strike,
                        "iv": iv,
                        "moneyness": moneyness
                    }
                    
                    if option["option_type"] == "call":
                        calls.append(option_info)
                    else:
                        puts.append(option_info)
                
                # Sort by strike
                calls = sorted(calls, key=lambda x: x["strike"])
                puts = sorted(puts, key=lambda x: x["strike"])
                
                # Find ATM option
                atm_calls = [c for c in calls if abs(c["moneyness"] - 1) < 0.05]
                atm_puts = [p for p in puts if abs(p["moneyness"] - 1) < 0.05]
                
                atm_call_iv = atm_calls[0]["iv"] if atm_calls else None
                atm_put_iv = atm_puts[0]["iv"] if atm_puts else None
                
                # Calculate ATM IV
                atm_iv = (atm_call_iv + atm_put_iv) / 2 if atm_call_iv and atm_put_iv else (atm_call_iv or atm_put_iv)
                
                # Calculate skew metrics (25-delta risk reversal)
                otm_calls = [c for c in calls if c["moneyness"] > 1.05]
                otm_puts = [p for p in puts if p["moneyness"] < 0.95]
                
                # 25-delta approximation (can be refined with actual delta calculation)
                call_25d = otm_calls[len(otm_calls)//4]["iv"] if otm_calls else None
                put_25d = otm_puts[len(otm_puts)//4]["iv"] if otm_puts else None
                
                risk_reversal = put_25d - call_25d if put_25d and call_25d else None
                butterfly = ((put_25d + call_25d) / 2 - atm_iv) if put_25d and call_25d and atm_iv else None
                
                iv_data[expiry] = {
                    "days_to_expiry": days_to_expiry,
                    "atm_iv": atm_iv,
                    "call_chain": calls,
                    "put_chain": puts,
                    "risk_reversal": risk_reversal,
                    "butterfly": butterfly
                }
            
            # Calculate term structure
            expirations = sorted(iv_data.keys())
            term_structure = []
            
            for exp in expirations:
                if "atm_iv" in iv_data[exp] and iv_data[exp]["atm_iv"]:
                    term_structure.append({
                        "expiry": exp,
                        "days": iv_data[exp]["days_to_expiry"],
                        "atm_iv": iv_data[exp]["atm_iv"]
                    })
            
            # Calculate term structure slopes
            for i in range(1, len(term_structure)):
                days_diff = term_structure[i]["days"] - term_structure[i-1]["days"]
                iv_diff = term_structure[i]["atm_iv"] - term_structure[i-1]["atm_iv"]
                
                if days_diff > 0:
                    term_structure[i]["slope"] = iv_diff / days_diff
                else:
                    term_structure[i]["slope"] = 0
            
            return {
                "iv_by_expiry": iv_data,
                "term_structure": term_structure
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing IV surface: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_option_data(self, option_id):
        """Get the latest data for a specific option"""
        try:
            # Query for the latest option price data
            latest_data = self.db.market_data_collection.find_one(
                {"instrument_id": option_id},
                sort=[("timestamp", -1)]
            )
            
            return latest_data
            
        except Exception as e:
            self.logger.error(f"Error getting option data: {str(e)}")
            return None
    
    def _analyze_options_metrics(self, symbol, exchange, option_chains):
        """
        Analyze key options metrics for decision making
        """
        try:
            metrics = {}
            
            for expiry, options in option_chains.items():
                days_to_expiry = (expiry - datetime.now()).days
                
                # Skip if less than 1 day to expiry
                if days_to_expiry < 1:
                    continue
                
                options_data = []
                
                # Process each option
                for option in options:
                    # Get latest option data
                    option_data = self._get_option_data(option["_id"])
                    if not option_data:
                        continue
                    
                    # Add option metrics
                    options_data.append({
                        "option_id": option["_id"],
                        "strike": option["strike"],
                        "option_type": option["option_type"],
                        "expiry": expiry,
                        "days_to_expiry": days_to_expiry,
                        "price": option_data.get("close"),
                        "iv": option_data.get("iv"),
                        "delta": option_data.get("delta"),
                        "gamma": option_data.get("gamma"),
                        "theta": option_data.get("theta"),
                        "vega": option_data.get("vega"),
                        "open_interest": option_data.get("open_interest"),
                        "volume": option_data.get("volume")
                    })
                
                # Calculate metrics by option type
                calls = [o for o in options_data if o["option_type"] == "call"]
                puts = [o for o in options_data if o["option_type"] == "put"]
                
                # Sort by strike
                calls = sorted(calls, key=lambda x: x["strike"])
                puts = sorted(puts, key=lambda x: x["strike"])
                
                # Find options with highest gamma (most sensitive to price changes)
                highest_gamma_call = max(calls, key=lambda x: x.get("gamma", 0) or 0) if calls else None
                highest_gamma_put = max(puts, key=lambda x: x.get("gamma", 0) or 0) if puts else None
                
                # Find options with highest theta (fastest time decay)
                highest_theta_call = min(calls, key=lambda x: x.get("theta", 0) or 0) if calls else None
                highest_theta_put = min(puts, key=lambda x: x.get("theta", 0) or 0) if puts else None
                
                # Find options with highest vega (most sensitive to volatility)
                highest_vega_call = max(calls, key=lambda x: x.get("vega", 0) or 0) if calls else None
                highest_vega_put = max(puts, key=lambda x: x.get("vega", 0) or 0) if puts else None
                
                # Find options with unusual volume/OI ratio
                vol_oi_calls = [c for c in calls if c.get("open_interest") and c.get("volume")]
                vol_oi_puts = [p for p in puts if p.get("open_interest") and p.get("volume")]
                
                highest_vol_oi_call = max(vol_oi_calls, key=lambda x: x.get("volume", 0) / x.get("open_interest", 1) if x.get("open_interest") else 0) if vol_oi_calls else None
                highest_vol_oi_put = max(vol_oi_puts, key=lambda x: x.get("volume", 0) / x.get("open_interest", 1) if x.get("open_interest") else 0) if vol_oi_puts else None
                
                metrics[expiry] = {
                    "days_to_expiry": days_to_expiry,
                    "highest_gamma": {
                        "call": {
                            "strike": highest_gamma_call["strike"] if highest_gamma_call else None,
                            "gamma": highest_gamma_call["gamma"] if highest_gamma_call else None
                        },
                        "put": {
                            "strike": highest_gamma_put["strike"] if highest_gamma_put else None,
                            "gamma": highest_gamma_put["gamma"] if highest_gamma_put else None
                        }
                    },
                    "highest_theta": {
                        "call": {
                            "strike": highest_theta_call["strike"] if highest_theta_call else None,
                            "theta": highest_theta_call["theta"] if highest_theta_call else None
                        },
                        "put": {
                            "strike": highest_theta_put["strike"] if highest_theta_put else None,
                            "theta": highest_theta_put["theta"] if highest_theta_put else None
                        }
                    },
                    "highest_vega": {
                        "call": {
                            "strike": highest_vega_call["strike"] if highest_vega_call else None,
                            "vega": highest_vega_call["vega"] if highest_vega_call else None
                        },
                        "put": {
                            "strike": highest_vega_put["strike"] if highest_vega_put else None,
                            "vega": highest_vega_put["vega"] if highest_vega_put else None
                        }
                    },
                    "unusual_volume_oi": {
                        "call": {
                            "strike": highest_vol_oi_call["strike"] if highest_vol_oi_call else None,
                            "volume": highest_vol_oi_call["volume"] if highest_vol_oi_call else None,
                            "open_interest": highest_vol_oi_call["open_interest"] if highest_vol_oi_call else None,
                            "ratio": highest_vol_oi_call["volume"] / highest_vol_oi_call["open_interest"] if highest_vol_oi_call and highest_vol_oi_call["open_interest"] else None
                        },
                        "put": {
                            "strike": highest_vol_oi_put["strike"] if highest_vol_oi_put else None,
                            "volume": highest_vol_oi_put["volume"] if highest_vol_oi_put else None,
                            "open_interest": highest_vol_oi_put["open_interest"] if highest_vol_oi_put else None,
                            "ratio": highest_vol_oi_put["volume"] / highest_vol_oi_put["open_interest"] if highest_vol_oi_put and highest_vol_oi_put["open_interest"] else None
                        }
                    }
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing options metrics: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_spread_opportunities(self, symbol, exchange, option_chains, underlying_price):
        """
        Analyze potential option spread opportunities
        """
        try:
            spreads = {}
            
            for expiry, options in option_chains.items():
                days_to_expiry = (expiry - datetime.now()).days
                
                # Skip if less than a week to expiry
                if days_to_expiry < 7:
                    continue
                
                calls = []
                puts = []
                
                # Process each option
                for option in options:
                    # Get latest option data
                    option_data = self._get_option_data(option["_id"])
                    if not option_data:
                        continue
                    
                    option_info = {
                        "strike": option["strike"],
                        "option_type": option["option_type"],
                        "price": option_data.get("close"),
                        "iv": option_data.get("iv"),
                        "delta": option_data.get("delta"),
                        "gamma": option_data.get("gamma"),
                        "theta": option_data.get("theta"),
                        "vega": option_data.get("vega")
                    }
                    
                    if option["option_type"] == "call":
                        calls.append(option_info)
                    else:
                        puts.append(option_info)
                
                # Sort by strike
                calls = sorted(calls, key=lambda x: x["strike"])
                puts = sorted(puts, key=lambda x: x["strike"])
                
                # Find vertical spreads (bull call, bear put)
                bull_call_spreads = []
                for i in range(len(calls) - 1):
                    if "price" not in calls[i] or "price" not in calls[i+1]:
                        continue
                        
                    long_call = calls[i]
                    short_call = calls[i+1]
                    
                    spread_cost = long_call["price"] - short_call["price"]
                    max_profit = short_call["strike"] - long_call["strike"] - spread_cost
                    risk_reward = max_profit / spread_cost if spread_cost > 0 else 0
                    
                    if risk_reward > 1.5:  # Good risk-reward
                        bull_call_spreads.append({
                            "long_strike": long_call["strike"],
                            "short_strike": short_call["strike"],
                            "cost": spread_cost,
                            "max_profit": max_profit,
                            "risk_reward": risk_reward
                        })
                
                # Bear put spreads
                bear_put_spreads = []
                for i in range(len(puts) - 1):
                    if "price" not in puts[i] or "price" not in puts[i+1]:
                        continue
                        
                    long_put = puts[i+1]  # Higher strike
                    short_put = puts[i]   # Lower strike
                    
                    spread_cost = long_put["price"] - short_put["price"]
                    max_profit = long_put["strike"] - short_put["strike"] - spread_cost
                    risk_reward = max_profit / spread_cost if spread_cost > 0 else 0
                    
                    if risk_reward > 1.5:  # Good risk-reward
                        bear_put_spreads.append({
                            "long_strike": long_put["strike"],
                            "short_strike": short_put["strike"],
                            "cost": spread_cost,
                            "max_profit": max_profit,
                            "risk_reward": risk_reward
                        })
                
                # Iron condors (combine bull put and bear call spreads)
                iron_condors = []
                
                # Find ATM strike
                atm_index = next((i for i, c in enumerate(calls) if c["strike"] >= underlying_price), 0)
                
                # Look for suitable wings
                if atm_index > 1 and atm_index < len(calls) - 2:
                    # Bull put (lower) wing
                    bull_put_short = puts[atm_index - 1]  # Just below ATM
                    bull_put_long = puts[atm_index - 2]   # Further below
                    
                    # Bear call (upper) wing
                    bear_call_short = calls[atm_index]      # Just above ATM
                    bear_call_long = calls[atm_index + 1]   # Further above
                    
                    if all(k in bull_put_short for k in ["price", "strike"]) and \
                       all(k in bull_put_long for k in ["price", "strike"]) and \
                       all(k in bear_call_short for k in ["price", "strike"]) and \
                       all(k in bear_call_long for k in ["price", "strike"]):
                        
                        # Credit received
                        credit = (bull_put_short["price"] - bull_put_long["price"]) + \
                                (bear_call_short["price"] - bear_call_long["price"])
                        
                        # Max loss (difference between long and short strikes minus credit)
                        wing_width = min(
                            bull_put_short["strike"] - bull_put_long["strike"],
                            bear_call_long["strike"] - bear_call_short["strike"]
                        )
                        max_loss = wing_width - credit
                        
                        # Risk-reward ratio
                        risk_reward = credit / max_loss if max_loss > 0 else 0
                        
                        if risk_reward > 0.25:  # Conservative filter
                            iron_condors.append({
                                "put_short_strike": bull_put_short["strike"],
                                "put_long_strike": bull_put_long["strike"],
                                "call_short_strike": bear_call_short["strike"],
                                "call_long_strike": bear_call_long["strike"],
                                "credit": credit,
                                "max_loss": max_loss,
                                "risk_reward": risk_reward
                            })
                
                # Calendar spreads (if we have multiple expiries)
                calendar_spreads = []
                
                # For now, just placeholder - would need data from different expirations
                
                spreads[expiry] = {
                    "days_to_expiry": days_to_expiry,
                    "bull_call_spreads": sorted(bull_call_spreads, key=lambda x: x["risk_reward"], reverse=True)[:3],
                    "bear_put_spreads": sorted(bear_put_spreads, key=lambda x: x["risk_reward"], reverse=True)[:3],
                    "iron_condors": sorted(iron_condors, key=lambda x: x["risk_reward"], reverse=True)[:3],
                    "calendar_spreads": calendar_spreads
                }
            
            return spreads
            
        except Exception as e:
            self.logger.error(f"Error analyzing spread opportunities: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_volume_oi(self, symbol, exchange, option_chains):
        """
        Analyze option volume and open interest patterns
        """
        try:
            volume_oi = {}
            
            for expiry, options in option_chains.items():
                call_volume = 0
                put_volume = 0
                call_oi = 0
                put_oi = 0
                
                # Process each option
                for option in options:
                    # Get latest option data
                    option_data = self._get_option_data(option["_id"])
                    if not option_data:
                        continue
                    
                    volume = option_data.get("volume", 0) or 0
                    oi = option_data.get("open_interest", 0) or 0
                    
                    if option["option_type"] == "call":
                        call_volume += volume
                        call_oi += oi
                    else:
                        put_volume += volume
                        put_oi += oi
                
                # Calculate ratios
                volume_ratio = put_volume / call_volume if call_volume > 0 else 0
                oi_ratio = put_oi / call_oi if call_oi > 0 else 0
                
                # Find strikes with highest volume and OI
                highest_call_volume = None
                highest_put_volume = None
                highest_call_oi = None
                highest_put_oi = None
                
                for option in options:
                    option_data = self._get_option_data(option["_id"])
                    if not option_data:
                        continue
                    
                    volume = option_data.get("volume", 0) or 0
                    oi = option_data.get("open_interest", 0) or 0
                    
                    if option["option_type"] == "call":
                        if highest_call_volume is None or volume > highest_call_volume.get("volume", 0):
                            highest_call_volume = {
                                "strike": option["strike"],
                                "volume": volume
                            }
                        
                        if highest_call_oi is None or oi > highest_call_oi.get("oi", 0):
                            highest_call_oi = {
                                "strike": option["strike"],
                                "oi": oi
                            }
                    else:
                        if highest_put_volume is None or volume > highest_put_volume.get("volume", 0):
                            highest_put_volume = {
                                "strike": option["strike"],
                                "volume": volume
                            }
                        
                        if highest_put_oi is None or oi > highest_put_oi.get("oi", 0):
                            highest_put_oi = {
                                "strike": option["strike"],
                                "oi": oi
                            }
                
                # Construct result
                volume_oi[expiry] = {
                    "call_volume_total": call_volume,
                    "put_volume_total": put_volume,
                    "call_oi_total": call_oi,
                    "put_oi_total": put_oi,
                    "put_call_volume_ratio": volume_ratio,
                    "put_call_oi_ratio": oi_ratio,
                    "highest_volume": {
                        "call": highest_call_volume,
                        "put": highest_put_volume
                    },
                    "highest_oi": {
                        "call": highest_call_oi,
                        "put": highest_put_oi
                    }
                }
            
            return volume_oi
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume and OI: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_put_call_ratio(self, symbol, exchange, option_chains):
        """
        Analyze put-call ratio across different metrics
        """
        try:
            # Historical PCR data retrieval
            pcr_history = self._get_pcr_history(symbol, exchange)
            
            # Current PCR data
            current_pcr = {}
            
            for expiry, options in option_chains.items():
                volume_data = self._analyze_volume_oi(symbol, exchange, {expiry: options})
                
                if expiry in volume_data:
                    expiry_data = volume_data[expiry]
                    
                    current_pcr[expiry] = {
                        "volume_ratio": expiry_data["put_call_volume_ratio"],
                        "oi_ratio": expiry_data["put_call_oi_ratio"]
                    }
            
            # Calculate PCR percentiles
            if pcr_history:
                volume_percentile = 0
                oi_percentile = 0
                
                # Current near-term PCR values
                nearest_expiry = min(current_pcr.keys())
                current_volume_pcr = current_pcr[nearest_expiry]["volume_ratio"]
                current_oi_pcr = current_pcr[nearest_expiry]["oi_ratio"]
                
                # Calculate percentiles
                volume_values = [h["volume_ratio"] for h in pcr_history]
                oi_values = [h["oi_ratio"] for h in pcr_history]
                
                volume_below = sum(1 for v in volume_values if v < current_volume_pcr)
                oi_below = sum(1 for v in oi_values if v < current_oi_pcr)
                
                volume_percentile = (volume_below / len(volume_values)) * 100 if volume_values else 0
                oi_percentile = (oi_below / len(oi_values)) * 100 if oi_values else 0
                
                # Market interpretation based on PCR
                sentiment = "Neutral"
                if current_oi_pcr > 1.2:
                    sentiment = "Bearish"
                elif current_oi_pcr < 0.8:
                    sentiment = "Bullish"
                
                # Contrarian signal based on extremes
                contrarian_signal = "None"
                if oi_percentile > 90:
                    contrarian_signal = "Bullish (contrarian)"
                elif oi_percentile < 10:
                    contrarian_signal = "Bearish (contrarian)"
                
                return {
                    "current_by_expiry": current_pcr,
                    "historical_data": pcr_history[-10:] if len(pcr_history) > 10 else pcr_history,
                    "current_near_term": {
                        "volume_ratio": current_volume_pcr,
                        "oi_ratio": current_oi_pcr,
                        "volume_percentile": volume_percentile,
                        "oi_percentile": oi_percentile
                    },
                    "sentiment": sentiment,
                    "contrarian_signal": contrarian_signal
                }
            
            return {
                "current_by_expiry": current_pcr,
                "historical_data": []
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing put-call ratio: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_pcr_history(self, symbol, exchange):
        """Get historical put-call ratio data"""
        try:
            # Query for historical PCR data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "data_type": "put_call_ratio"
            }
            
            # Get last 30 days of data
            history = list(self.db.market_metadata_collection.find(query).sort("date", -1).limit(30))
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting PCR history: {str(e)}")
            return []
    
    def _analyze_skew(self, symbol, exchange, option_chains, underlying_price):
        """
        Analyze options skew for trading insights
        """
        try:
            skew_analysis = {}
            
            for expiry, options in option_chains.items():
                days_to_expiry = (expiry - datetime.now()).days
                
                # Skip if less than 1 day to expiry
                if days_to_expiry < 1:
                    continue
                
                calls = []
                puts = []
                
                # Process each option
                for option in options:
                    # Get latest option data
                    option_data = self._get_option_data(option["_id"])
                    if not option_data or "iv" not in option_data:
                        continue
                    
                    strike = option["strike"]
                    iv = option_data["iv"]
                    
                    # Calculate normalized strike (strike / current price)
                    normalized_strike = strike / underlying_price
                    
                    option_info = {
                        "strike": strike,
                        "normalized_strike": normalized_strike,
                        "iv": iv
                    }
                    
                    if option["option_type"] == "call":
                        calls.append(option_info)
                    else:
                        puts.append(option_info)
                
                # Sort by strike
                calls = sorted(calls, key=lambda x: x["strike"])
                puts = sorted(puts, key=lambda x: x["strike"])
                
                # Find ATM IV
                atm_calls = [c for c in calls if abs(c["normalized_strike"] - 1) < 0.05]
                atm_puts = [p for p in puts if abs(p["normalized_strike"] - 1) < 0.05]
                
                atm_call_iv = atm_calls[0]["iv"] if atm_calls else None
                atm_put_iv = atm_puts[0]["iv"] if atm_puts else None
                
                atm_iv = (atm_call_iv + atm_put_iv) / 2 if atm_call_iv and atm_put_iv else (atm_call_iv or atm_put_iv)
                
                # Skew calculation - compare OTM puts to ATM
                otm_puts = [p for p in puts if p["normalized_strike"] < 0.9]  # 10% OTM
                otm_calls = [c for c in calls if c["normalized_strike"] > 1.1]  # 10% OTM
                
                if otm_puts and atm_iv:
                    put_skew = sum(p["iv"] for p in otm_puts) / len(otm_puts) - atm_iv
                else:
                    put_skew = None
                
                if otm_calls and atm_iv:
                    call_skew = sum(c["iv"] for c in otm_calls) / len(otm_calls) - atm_iv
                else:
                    call_skew = None
                
                # Skew ratio (put skew / call skew)
                # Skew ratio (put skew / call skew)
                skew_ratio = abs(put_skew / call_skew) if put_skew and call_skew and call_skew != 0 else None
                
                # Skew interpretation
                interpretation = "Neutral"
                
                if put_skew and put_skew > 0.05:
                    interpretation = "Negative Skew (Tail Risk Concern)"
                    if skew_ratio and skew_ratio > 2:
                        interpretation = "Strong Negative Skew (High Tail Risk Concern)"
                elif call_skew and call_skew > 0.03:
                    interpretation = "Positive Skew (Upside Momentum Expected)"
                
                # Volatility smile or smirk?
                smile_type = "Flat"
                if put_skew and call_skew:
                    if put_skew > 0.02 and call_skew > 0.02:
                        smile_type = "Smile"
                    elif put_skew > 0.02 and call_skew < 0.01:
                        smile_type = "Put Smirk"
                    elif call_skew > 0.02 and put_skew < 0.01:
                        smile_type = "Call Smirk"
                
                skew_analysis[expiry] = {
                    "days_to_expiry": days_to_expiry,
                    "atm_iv": atm_iv,
                    "put_skew": put_skew,
                    "call_skew": call_skew,
                    "skew_ratio": skew_ratio,
                    "smile_type": smile_type,
                    "interpretation": interpretation
                }
            
            return skew_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing skew: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _generate_signals(self, analysis_result):
        """
        Generate trading signals based on options analysis
        """
        try:
            signals = []
            
            # Extract analysis components
            iv_surface = analysis_result["options_analysis"]["iv_surface"]
            metrics = analysis_result["options_analysis"]["options_metrics"]
            spreads = analysis_result["options_analysis"]["spread_opportunities"]
            volume_oi = analysis_result["options_analysis"]["volume_oi_analysis"]
            pcr = analysis_result["options_analysis"]["put_call_ratio"]
            skew = analysis_result["options_analysis"]["skew_analysis"]
            
            # Process nearest expiry
            if not spreads:
                return signals
                
            nearest_expiry = min(spreads.keys())
            
            # 1. Directional signals from PCR
            if isinstance(pcr, dict) and "sentiment" in pcr:
                if pcr["sentiment"] == "Bullish":
                    signals.append({
                        "strategy": "Directional (PCR)",
                        "signal": "Bullish - Consider Long Calls or Bull Call Spreads",
                        "strength": 3,
                        "reason": f"Put-call ratio indicates bullish sentiment with PCR at {pcr['current_near_term']['oi_ratio']:.2f}"
                    })
                elif pcr["sentiment"] == "Bearish":
                    signals.append({
                        "strategy": "Directional (PCR)",
                        "signal": "Bearish - Consider Long Puts or Bear Put Spreads",
                        "strength": 3,
                        "reason": f"Put-call ratio indicates bearish sentiment with PCR at {pcr['current_near_term']['oi_ratio']:.2f}"
                    })
            
            # 2. Contrarian signals from PCR extremes
            if isinstance(pcr, dict) and "contrarian_signal" in pcr and pcr["contrarian_signal"] != "None":
                signals.append({
                    "strategy": "Contrarian (PCR)",
                    "signal": pcr["contrarian_signal"],
                    "strength": 4,
                    "reason": f"Extreme put-call ratio at {pcr['current_near_term']['oi_percentile']:.0f}th percentile suggests potential reversal"
                })
            
            # 3. Volatility signals from IV surface
            if nearest_expiry in skew:
                expiry_skew = skew[nearest_expiry]
                if "interpretation" in expiry_skew:
                    if "Negative Skew" in expiry_skew["interpretation"]:
                        signals.append({
                            "strategy": "Volatility (Skew)",
                            "signal": "Consider Long Put Protection or Put Spreads",
                            "strength": 3,
                            "reason": f"{expiry_skew['interpretation']} indicates market concern about downside risk"
                        })
                    elif "Positive Skew" in expiry_skew["interpretation"]:
                        signals.append({
                            "strategy": "Volatility (Skew)",
                            "signal": "Consider Call Backspread for Upside Capture",
                            "strength": 3,
                            "reason": f"{expiry_skew['interpretation']} indicates expectations of upside momentum"
                        })
            
            # 4. Term structure signals
            if "term_structure" in iv_surface and len(iv_surface["term_structure"]) > 1:
                last_slope = iv_surface["term_structure"][-1].get("slope")
                if last_slope is not None:
                    if last_slope > 0.002:  # Steep upward slope
                        signals.append({
                            "strategy": "Volatility (Term Structure)",
                            "signal": "Consider Calendar Spreads (Sell Front-Month, Buy Back-Month)",
                            "strength": 3,
                            "reason": f"Upward sloping volatility term structure with slope of {last_slope:.4f}"
                        })
                    elif last_slope < -0.002:  # Steep downward slope
                        signals.append({
                            "strategy": "Volatility (Term Structure)",
                            "signal": "Consider Reverse Calendar Spreads (Buy Front-Month, Sell Back-Month)",
                            "strength": 3,
                            "reason": f"Downward sloping volatility term structure with slope of {last_slope:.4f}"
                        })
            
            # 5. Spread opportunities
            if nearest_expiry in spreads:
                expiry_spreads = spreads[nearest_expiry]
                
                # Bull call spreads
                if expiry_spreads["bull_call_spreads"]:
                    best_bull_call = expiry_spreads["bull_call_spreads"][0]
                    signals.append({
                        "strategy": "Vertical Spread",
                        "signal": f"Bull Call Spread: Long {best_bull_call['long_strike']} / Short {best_bull_call['short_strike']}",
                        "strength": min(int(best_bull_call["risk_reward"] * 2), 5),
                        "reason": f"Attractive risk-reward of {best_bull_call['risk_reward']:.2f} with max profit of {best_bull_call['max_profit']:.2f}"
                    })
                
                # Bear put spreads
                if expiry_spreads["bear_put_spreads"]:
                    best_bear_put = expiry_spreads["bear_put_spreads"][0]
                    signals.append({
                        "strategy": "Vertical Spread",
                        "signal": f"Bear Put Spread: Long {best_bear_put['long_strike']} / Short {best_bear_put['short_strike']}",
                        "strength": min(int(best_bear_put["risk_reward"] * 2), 5),
                        "reason": f"Attractive risk-reward of {best_bear_put['risk_reward']:.2f} with max profit of {best_bear_put['max_profit']:.2f}"
                    })
                
                # Iron condors
                if expiry_spreads["iron_condors"]:
                    best_iron_condor = expiry_spreads["iron_condors"][0]
                    signals.append({
                        "strategy": "Iron Condor",
                        "signal": f"Iron Condor: Put {best_iron_condor['put_long_strike']}/{best_iron_condor['put_short_strike']} - Call {best_iron_condor['call_short_strike']}/{best_iron_condor['call_long_strike']}",
                        "strength": min(int(best_iron_condor["risk_reward"] * 5), 4),
                        "reason": f"Credit of {best_iron_condor['credit']:.2f} with risk-reward of {best_iron_condor['risk_reward']:.2f}"
                    })
            
            # 6. Unusual activity signals
            for exp, vol_data in volume_oi.items():
                # Unusual call activity
                if "highest_volume" in vol_data and vol_data["highest_volume"]["call"]:
                    call_strike = vol_data["highest_volume"]["call"]["strike"]
                    call_volume = vol_data["highest_volume"]["call"]["volume"]
                    
                    if call_volume > 3 * vol_data["call_volume_total"] / len(options_chains[exp]):
                        signals.append({
                            "strategy": "Unusual Activity",
                            "signal": f"Unusual Call Activity at Strike {call_strike}",
                            "strength": 2,
                            "reason": f"High volume of {call_volume} contracts suggests potential bullish positioning"
                        })
                
                # Unusual put activity
                if "highest_volume" in vol_data and vol_data["highest_volume"]["put"]:
                    put_strike = vol_data["highest_volume"]["put"]["strike"]
                    put_volume = vol_data["highest_volume"]["put"]["volume"]
                    
                    if put_volume > 3 * vol_data["put_volume_total"] / len(options_chains[exp]):
                        signals.append({
                            "strategy": "Unusual Activity",
                            "signal": f"Unusual Put Activity at Strike {put_strike}",
                            "strength": 2,
                            "reason": f"High volume of {put_volume} contracts suggests potential bearish positioning or hedging"
                        })
            
            # Filter and sort signals by strength
            filtered_signals = sorted(signals, key=lambda x: x.get("strength", 0), reverse=True)
            
            return filtered_signals[:5]  # Return top 5 signals
            
        except Exception as e:
            self.logger.error(f"Error generating options signals: {str(e)}")
            return []
    
    def _save_analysis(self, analysis_result):
        """Save analysis results to database"""
        try:
            # Create document for database
            doc = {
                "symbol": analysis_result["symbol"],
                "exchange": analysis_result["exchange"],
                "analysis_type": "options",
                "timestamp": datetime.now(),
                "data": analysis_result
            }
            
            # Insert or update in database
            self.db.analysis_collection.replace_one(
                {
                    "symbol": analysis_result["symbol"],
                    "exchange": analysis_result["exchange"],
                    "analysis_type": "options"
                },
                doc,
                upsert=True
            )
            
        except Exception as e:
            self.logger.error(f"Error saving analysis: {str(e)}")