# futures.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

class FuturesTrader:
    def __init__(self, db_connector):
        """Initialize the futures trader"""
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
    
    def analyze_trading_opportunity(self, symbol, exchange, expiry=None, prediction=None, timeframe="day"):
        """
        Analyze potential trading opportunities for futures
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - expiry: Optional specific expiry date to analyze
        - prediction: Optional prediction data
        - timeframe: Timeframe for analysis
        
        Returns:
        - Dictionary with trading opportunity details
        """
        try:
            # Get available futures contracts for this symbol
            contracts = self._get_futures_contracts(symbol, exchange, expiry)
            if not contracts:
                return {"status": "error", "message": "No futures contracts available"}
            
            # If no specific expiry, use the nearest one
            if not expiry:
                contract = contracts[0]  # Nearest expiry
                expiry = contract["expiry"]
            else:
                contract = next((c for c in contracts if c["expiry"] == expiry), None)
                if not contract:
                    return {"status": "error", "message": f"No contract found for expiry {expiry}"}
            
            # Get futures market data
            futures_data = self._get_futures_data(symbol, exchange, expiry, timeframe)
            if not futures_data or len(futures_data) < 20:
                return {"status": "error", "message": "Insufficient futures market data"}
            
            # Get spot data for basis analysis
            spot_data = self._get_spot_data(symbol, exchange, timeframe)
            
            # Get technical analysis
            technical = self._get_technical_analysis(symbol, exchange, timeframe, instrument_type="futures", expiry=expiry)
            
            # Get market context and sentiment
            market_context = self._get_market_context(symbol, exchange)
            
            # Analyze basis and term structure
            basis_analysis = self._analyze_basis(futures_data, spot_data)
            term_structure = self._analyze_term_structure(symbol, exchange, contracts)
            
            # Analyze futures-specific patterns
            roll_analysis = self._analyze_roll_opportunity(contract)
            open_interest = self._analyze_open_interest(symbol, exchange, expiry)
            
            # Determine trading parameters
            trade_params = self._determine_trading_parameters(
                symbol, exchange, expiry, futures_data, technical, 
                market_context, basis_analysis, term_structure, 
                roll_analysis, open_interest, prediction
            )
            
            # Filter out unsuitable opportunities
            if not trade_params["suitable_for_trading"]:
                return {
                    "status": "success",
                    "trade_opportunity": False,
                    "reason": trade_params["reason"]
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
                "direction": trade_params["direction"],
                "strategy": trade_params["strategy"],
                "timeframe": timeframe,
                "entry": {
                    "type": trade_params["entry_type"],
                    "price": trade_params["entry_price"],
                    "valid_until": trade_params["entry_valid_until"]
                },
                "exit": {
                    "stop_loss": trade_params["stop_loss"],
                    "take_profit": trade_params["take_profit"],
                    "trailing_stop": trade_params["trailing_stop"]
                },
                "risk_reward": trade_params["risk_reward"],
                "position_size": trade_params["position_size"],
                "confidence": trade_params["confidence"],
                "technical_signals": technical["signals"] if technical else None,
                "basis_analysis": basis_analysis,
                "term_structure": term_structure,
                "roll_analysis": roll_analysis,
                "open_interest": open_interest,
                "market_context": market_context["summary"] if market_context else None
            }
            
            # Save opportunity to database
            self._save_opportunity(opportunity)
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error analyzing futures opportunity: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_futures_contracts(self, symbol, exchange, expiry=None):
        """Get available futures contracts for a symbol"""
        try:
            # Base query for futures contracts
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": "futures",
                "status": "active"
            }
            
            # Add expiry filter if provided
            if expiry:
                query["expiry"] = expiry
            
            # Get contracts sorted by expiry (nearest first)
            contracts = list(self.db.instrument_collection.find(query).sort("expiry", 1))
            
            return contracts
            
        except Exception as e:
            self.logger.error(f"Error getting futures contracts: {str(e)}")
            return []
    
    def _get_futures_data(self, symbol, exchange, expiry, timeframe):
        """Get market data for a futures contract"""
        try:
            # Query for futures data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": "futures",
                "expiry": expiry,
                "timeframe": timeframe
            }
            
            # Get the last 100 data points
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "open_interest": 1}
            ).sort("timestamp", -1).limit(100))
            
            if not data:
                return None
                
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting futures data: {str(e)}")
            return None
    
    def _get_spot_data(self, symbol, exchange, timeframe):
        """Get market data for the underlying spot"""
        try:
            # Query for spot data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": "equity",
                "timeframe": timeframe
            }
            
            # Get the last 100 data points
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "close": 1}
            ).sort("timestamp", -1).limit(100))
            
            if not data:
                return None
                
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting spot data: {str(e)}")
            return None
    
    def _get_technical_analysis(self, symbol, exchange, timeframe, instrument_type="futures", expiry=None):
        """Get technical analysis for futures contract"""
        try:
            # Try to get from TechnicalAnalyzer
            try:
                from research.technical_analyzer import TechnicalAnalyzer
                analyzer = TechnicalAnalyzer(self.db)
                analysis = analyzer.analyze(
                    symbol, exchange, timeframe, 
                    instrument_type=instrument_type, 
                    expiry=expiry
                )
                return analysis
            except Exception as tech_error:
                self.logger.warning(f"Technical analyzer not available: {str(tech_error)}")
            
            # Fallback to database query
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "analysis_type": "technical",
                "instrument_type": instrument_type
            }
            
            if expiry:
                query["expiry"] = expiry
            
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
    
    def _analyze_basis(self, futures_data, spot_data):
        """Analyze basis between futures and spot"""
        try:
            if futures_data is None or spot_data is None:
                return {"status": "error", "message": "Missing data for basis analysis"}
            
            # Merge futures and spot data on timestamp
            merged = pd.merge(
                futures_data[["timestamp", "close"]], 
                spot_data[["timestamp", "close"]], 
                on="timestamp", 
                suffixes=("_futures", "_spot")
            )
            
            if len(merged) < 5:
                return {"status": "error", "message": "Insufficient overlapping data points"}
            
            # Calculate basis and basis percentage
            merged["basis"] = merged["close_futures"] - merged["close_spot"]
            merged["basis_percent"] = (merged["basis"] / merged["close_spot"]) * 100
            
            # Calculate basis statistics
            current_basis = merged["basis"].iloc[-1]
            current_basis_percent = merged["basis_percent"].iloc[-1]
            mean_basis = merged["basis"].mean()
            std_basis = merged["basis"].std()
            z_score = (current_basis - mean_basis) / std_basis if std_basis > 0 else 0
            
            # Check basis trend
            basis_5d_ago = merged["basis"].iloc[-6] if len(merged) > 5 else merged["basis"].iloc[0]
            basis_trend = "widening" if current_basis > basis_5d_ago else "narrowing"
            
            # Interpret basis
            if z_score > 2:
                interpretation = "Unusually wide basis - potential for convergence"
                trading_signal = "Consider short futures, long spot arbitrage"
            elif z_score < -2:
                interpretation = "Unusually narrow basis - potential for divergence"
                trading_signal = "Consider long futures, short spot arbitrage"
            else:
                interpretation = "Basis within normal range"
                trading_signal = "No clear basis trading opportunity"
            
            return {
                "status": "success",
                "current_basis": current_basis,
                "current_basis_percent": current_basis_percent,
                "mean_basis": mean_basis,
                "basis_z_score": z_score,
                "basis_trend": basis_trend,
                "interpretation": interpretation,
                "trading_signal": trading_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing basis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_term_structure(self, symbol, exchange, contracts):
        """Analyze futures term structure"""
        try:
            if not contracts or len(contracts) < 2:
                return {"status": "error", "message": "Insufficient contracts for term structure analysis"}
            
            # Extract expiry dates and prices
            term_structure = []
            for contract in contracts:
                # Get latest price
                latest_price = self.db.market_data_collection.find_one(
                    {
                        "symbol": symbol,
                        "exchange": exchange,
                        "instrument_type": "futures",
                        "expiry": contract["expiry"]
                    },
                    {"close": 1}
                )
                
                if latest_price and "close" in latest_price:
                    term_structure.append({
                        "expiry": contract["expiry"],
                        "price": latest_price["close"],
                        "days_to_expiry": (contract["expiry"] - datetime.now()).days
                    })
            
            if len(term_structure) < 2:
                return {"status": "error", "message": "Insufficient price data for term structure analysis"}
            
            # Sort by days to expiry
            term_structure = sorted(term_structure, key=lambda x: x["days_to_expiry"])
            
            # Calculate term structure shape
            slopes = []
            for i in range(1, len(term_structure)):
                days_diff = term_structure[i]["days_to_expiry"] - term_structure[i-1]["days_to_expiry"]
                price_diff = term_structure[i]["price"] - term_structure[i-1]["price"]
                
                if days_diff > 0:
                    # Annualized slope in percentage
                    slope = (price_diff / term_structure[i-1]["price"]) * (365 / days_diff) * 100
                    slopes.append({
                        "from_expiry": term_structure[i-1]["expiry"],
                        "to_expiry": term_structure[i]["expiry"],
                        "slope": slope
                    })
            
            # Determine term structure shape
            if all(s["slope"] > 0 for s in slopes):
                shape = "contango"
                interpretation = "Market expects higher prices in the future (bearish)"
            elif all(s["slope"] < 0 for s in slopes):
                shape = "backwardation"
                interpretation = "Market expects lower prices in the future (bullish)"
            elif slopes[0]["slope"] > 0 and any(s["slope"] < 0 for s in slopes[1:]):
                shape = "humped"
                interpretation = "Mixed expectations with near-term strength"
            elif slopes[0]["slope"] < 0 and any(s["slope"] > 0 for s in slopes[1:]):
                shape = "inverted_humped"
                interpretation = "Mixed expectations with near-term weakness"
            else:
                shape = "mixed"
                interpretation = "Unclear market expectations"
            
            # Trading opportunities based on term structure
            if shape == "contango":
                trading_signal = "Calendar spread: Short near-term, long far-term"
            elif shape == "backwardation":
                trading_signal = "Calendar spread: Long near-term, short far-term"
            elif shape == "humped":
                trading_signal = "Butterfly spread: Long near and far terms, short mid-term"
            elif shape == "inverted_humped":
                trading_signal = "Reverse butterfly spread: Short near and far terms, long mid-term"
            else:
                trading_signal = "No clear term structure opportunity"
            
            return {
                "status": "success",
                "contracts": term_structure,
                "slopes": slopes,
                "shape": shape,
                "interpretation": interpretation,
                "trading_signal": trading_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing term structure: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_roll_opportunity(self, contract):
        """Analyze rollover opportunity as expiry approaches"""
        try:
            expiry = contract["expiry"]
            days_to_expiry = (expiry - datetime.now()).days
            
            if days_to_expiry > 15:
                return {
                    "status": "success",
                    "days_to_expiry": days_to_expiry,
                    "roll_required": False,
                    "urgency": "none",
                    "recommendation": "Hold current contract"
                }
            
            # Get next contract if available
            symbol = contract["symbol"]
            exchange = contract["exchange"]
            
            next_contract = self.db.instrument_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": "futures",
                "status": "active",
                "expiry": {"$gt": expiry}
            }, sort=[("expiry", 1)])
            
            if not next_contract:
                return {
                    "status": "success",
                    "days_to_expiry": days_to_expiry,
                    "roll_required": False,
                    "urgency": "none",
                    "recommendation": "No next contract available"
                }
            
            # Get prices for current and next contract
            current_price = self._get_contract_price(contract)
            next_price = self._get_contract_price(next_contract)
            
            if not current_price or not next_price:
                return {
                    "status": "success",
                    "days_to_expiry": days_to_expiry,
                    "roll_required": days_to_expiry <= 5,
                    "urgency": "high" if days_to_expiry <= 2 else "medium" if days_to_expiry <= 5 else "low",
                    "recommendation": "Price data unavailable, roll based on days to expiry"
                }
            
            # Calculate roll cost
            roll_cost = next_price - current_price
            roll_cost_percent = (roll_cost / current_price) * 100
            
            # Analyze roll efficiency based on days to expiry and roll cost
            if days_to_expiry <= 2:
                urgency = "critical"
                recommendation = "Immediate rollover required to avoid delivery"
            elif days_to_expiry <= 5:
                urgency = "high"
                if roll_cost_percent > 0.5:
                    recommendation = "Rollover soon, roll cost may increase further"
                else:
                    recommendation = "Rollover soon, favorable roll cost"
            elif days_to_expiry <= 10:
                urgency = "medium"
                if roll_cost_percent > 0.5:
                    recommendation = "Consider early rollover to minimize roll cost"
                else:
                    recommendation = "Monitor roll cost daily, no immediate action required"
            else:
                urgency = "low"
                recommendation = "Monitor roll cost, sufficient time before expiry"
            
            return {
                "status": "success",
                "days_to_expiry": days_to_expiry,
                "next_expiry": next_contract["expiry"],
                "days_between_expiries": (next_contract["expiry"] - expiry).days,
                "current_price": current_price,
                "next_price": next_price,
                "roll_cost": roll_cost,
                "roll_cost_percent": roll_cost_percent,
                "roll_required": days_to_expiry <= 5,
                "urgency": urgency,
                "recommendation": recommendation
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing roll opportunity: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_contract_price(self, contract):
        """Get latest price for a futures contract"""
        try:
            query = {
                "symbol": contract["symbol"],
                "exchange": contract["exchange"],
                "instrument_type": "futures",
                "expiry": contract["expiry"]
            }
            
            latest_data = self.db.market_data_collection.find_one(
                query,
                {"close": 1}
            )
            
            if latest_data and "close" in latest_data:
                return latest_data["close"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting contract price: {str(e)}")
            return None
    
    def _analyze_open_interest(self, symbol, exchange, expiry):
        """Analyze open interest patterns"""
        try:
            # Get futures data with OI
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": "futures",
                "expiry": expiry
            }
            
            # Get the last 20 data points
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "close": 1, "open_interest": 1, "volume": 1}
            ).sort("timestamp", -1).limit(20))
            
            if not data or len(data) < 5 or "open_interest" not in data[0]:
                return {"status": "error", "message": "Insufficient open interest data"}
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")
            
            # Calculate OI changes
            df["oi_change"] = df["open_interest"].diff()
            df["oi_change_percent"] = (df["oi_change"] / df["open_interest"].shift(1)) * 100
            df["price_change"] = df["close"].diff()
            
            # Price-OI analysis for last 5 days
            recent_df = df.tail(5)
            
            # Classify each day's price-OI relationship
            price_oi_relations = []
            for i in range(1, len(recent_df)):
                price_change = recent_df["price_change"].iloc[i]
                oi_change = recent_df["oi_change"].iloc[i]
                
                if price_change > 0 and oi_change > 0:
                    relation = "long_buildup"
                    interpretation = "Bullish (Long positions being built)"
                elif price_change < 0 and oi_change > 0:
                    relation = "short_buildup"
                    interpretation = "Bearish (Short positions being built)"
                elif price_change > 0 and oi_change < 0:
                    relation = "short_covering"
                    interpretation = "Bullish (Short positions being covered)"
                elif price_change < 0 and oi_change < 0:
                    relation = "long_unwinding"
                    interpretation = "Bearish (Long positions being unwound)"
                else:
                    relation = "neutral"
                    interpretation = "Neutral (No significant change)"
                
                price_oi_relations.append({
                    "date": recent_df["timestamp"].iloc[i],
                    "relation": relation,
                    "interpretation": interpretation,
                    "price_change": price_change,
                    "oi_change": oi_change
                })
            
            # Determine overall trend from most recent days
            recent_relations = [r["relation"] for r in price_oi_relations[-3:]]
            
            if recent_relations.count("long_buildup") >= 2 or recent_relations.count("short_covering") >= 2:
                overall_trend = "bullish"
                signal = "Bullish momentum based on open interest patterns"
            elif recent_relations.count("short_buildup") >= 2 or recent_relations.count("long_unwinding") >= 2:
                overall_trend = "bearish"
                signal = "Bearish momentum based on open interest patterns"
            else:
                overall_trend = "mixed"
                signal = "Mixed signals from open interest patterns"
            
            # Calculate OI momentum (rate of change)
            oi_momentum = recent_df["oi_change_percent"].mean()
            
            # Calculate volume-OI ratio
            if "volume" in recent_df.columns:
                vol_oi_ratio = recent_df["volume"].iloc[-1] / recent_df["open_interest"].iloc[-1]
            else:
                vol_oi_ratio = None
            
            return {
                "status": "success",
                "current_oi": df["open_interest"].iloc[-1],
                "oi_change_1d": df["oi_change"].iloc[-1],
                "oi_change_percent_1d": df["oi_change_percent"].iloc[-1],
                "oi_momentum": oi_momentum,
                "volume_oi_ratio": vol_oi_ratio,
                "daily_analysis": price_oi_relations,
                "overall_trend": overall_trend,
                "trading_signal": signal
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing open interest: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _determine_trading_parameters(self, symbol, exchange, expiry, futures_data, 
                                    technical, market_context, basis_analysis, 
                                    term_structure, roll_analysis, open_interest, prediction):
        """Determine key trading parameters for futures"""
        try:
            # Default parameters
            params = {
                "suitable_for_trading": False,
                "reason": "No compelling opportunity",
                "direction": None,
                "strategy": None,
                "entry_type": "market",
                "entry_price": None,
                "entry_valid_until": datetime.now() + timedelta(days=1),
                "stop_loss": None,
                "take_profit": None,
                "trailing_stop": False,
                "risk_reward": None,
                "position_size": None,
                "confidence": 0.0
            }
            
            # Check if nearing expiry
            days_to_expiry = (expiry - datetime.now()).days
            if days_to_expiry <= 2:
                params["reason"] = "Too close to expiry for new positions"
                return params
            
            # Current price
            current_price = futures_data["close"].iloc[-1] if futures_data is not None and len(futures_data) > 0 else None
            if not current_price:
                params["reason"] = "Unable to determine current price"
                return params
            
            # Get direction from prediction, technical, or open interest
            direction = None
            confidence = 0.5
            
            # First check prediction if available
            if prediction and "prediction" in prediction:
                direction = "long" if prediction["prediction"] == "up" else \
                           "short" if prediction["prediction"] == "down" else None
                confidence = prediction.get("confidence", 0.5)
            
            # If no prediction, check technical signals
            elif technical and "signals" in technical:
                # Count bullish vs bearish signals
                bullish_count = sum(1 for s in technical["signals"] if s.get("direction") == "bullish")
                bearish_count = sum(1 for s in technical["signals"] if s.get("direction") == "bearish")
                
                if bullish_count > bearish_count:
                    direction = "long"
                    confidence = min(0.5 + (bullish_count - bearish_count) * 0.05, 0.9)
                elif bearish_count > bullish_count:
                    direction = "short"
                    confidence = min(0.5 + (bearish_count - bullish_count) * 0.05, 0.9)
            
            # If still no direction, check open interest
            elif open_interest and "overall_trend" in open_interest:
                if open_interest["overall_trend"] == "bullish":
                    direction = "long"
                    confidence = 0.6
                elif open_interest["overall_trend"] == "bearish":
                    direction = "short"
                    confidence = 0.6
            
            # If no direction found, exit
            if not direction:
                params["reason"] = "No clear directional bias"
                return params
            
            # Check if near expiry and roll required
            if roll_analysis and roll_analysis.get("roll_required", False):
                params["reason"] = "Roll required, no new positions until rolled"
                return params
            
            # Check if basis indicates an arbitrage opportunity
            if basis_analysis and "trading_signal" in basis_analysis and "arbitrage" in basis_analysis["trading_signal"].lower():
                # Use basis strategy instead of directional
                strategy = "basis_arbitrage"
                
                # Direction based on basis arbitrage signal
                if "short futures" in basis_analysis["trading_signal"].lower():
                    direction = "short"
                elif "long futures" in basis_analysis["trading_signal"].lower():
                    direction = "long"
                
                confidence = min(0.5 + abs(basis_analysis.get("basis_z_score", 0)) * 0.1, 0.9)
            else:
                # Use regular directional strategy based on technical/prediction
                strategy = "directional"
            
            # Check if term structure indicates calendar spread
            if term_structure and "trading_signal" in term_structure and "calendar spread" in term_structure["trading_signal"].lower():
                if len(term_structure.get("contracts", [])) >= 2:
                    # This is a spread trade, handle differently
                    params["reason"] = "Calendar spread opportunity requires special handling"
                    return params
            
            # Determine entry type and price
            if technical and "support_resistance" in technical:
                sr_levels = technical["support_resistance"]
                
                if direction == "long":
                    # For longs, enter at support or slight pullback
                    supports = sr_levels.get("support", [])
                    nearest_support = max([s for s in supports if s < current_price], default=None)
                    
                    if nearest_support and (current_price - nearest_support) / current_price < 0.03:
                        # Near support, use limit order slightly below current price
                        entry_type = "limit"
                        entry_price = current_price * 0.995  # 0.5% below current price
                    else:
                        # Use market order
                        entry_type = "market"
                        entry_price = current_price
                
                elif direction == "short":
                    # For shorts, enter at resistance or slight rally
                    resistances = sr_levels.get("resistance", [])
                    nearest_resistance = min([r for r in resistances if r > current_price], default=None)
                    
                    if nearest_resistance and (nearest_resistance - current_price) / current_price < 0.03:
                        # Near resistance, use limit order slightly above current price
                        entry_type = "limit"
                        entry_price = current_price * 1.005  # 0.5% above current price
                    else:
                        # Use market order
                        entry_type = "market"
                        entry_price = current_price
            else:
                # Default to market order at current price
                entry_type = "market"
                entry_price = current_price
            
            # Determine stop loss
            stop_loss = None
            
            if technical and "support_resistance" in technical:
                sr_levels = technical["support_resistance"]
                
                if direction == "long":
                    # For longs, place stop below nearest support
                    supports = sr_levels.get("support", [])
                    nearest_support = max([s for s in supports if s < entry_price], default=None)
                    
                    if nearest_support:
                        stop_loss = nearest_support * 0.99  # 1% below support
                
                elif direction == "short":
                    # For shorts, place stop above nearest resistance
                    resistances = sr_levels.get("resistance", [])
                    nearest_resistance = min([r for r in resistances if r > entry_price], default=None)
                    
                    if nearest_resistance:
                        stop_loss = nearest_resistance * 1.01  # 1% above resistance
            
            # If no support/resistance stop available, use ATR-based stop
            if not stop_loss and technical and "indicators" in technical and "atr_14" in technical["indicators"]:
                atr = technical["indicators"]["atr_14"]
                
                if direction == "long":
                    stop_loss = entry_price - (atr * 2)
                else:
                    stop_loss = entry_price + (atr * 2)
            
            # Fallback to percentage-based stop
            if not stop_loss:
                if direction == "long":
                    stop_loss = entry_price * 0.97  # 3% below entry for futures (tighter than equity)
                else:
                    stop_loss = entry_price * 1.03  # 3% above entry for futures
            
            # Determine take profit based on risk-reward ratio
            risk = abs(entry_price - stop_loss)
            take_profit = entry_price + (risk * 2) if direction == "long" else entry_price - (risk * 2)
            risk_reward = 2.0  # Fixed 2:1 reward-to-risk ratio
            
            # Use trailing stop for strong trends
            trailing_stop = False
            if technical and "trend_strength" in technical:
                trend_strength = technical["trend_strength"]
                if trend_strength == "strong":
                    trailing_stop = True
            
            # Calculate position size (dummy value for now)
            position_size = 1  # 1 lot
            
            # Incorporate market context
            if market_context and "market_regime" in market_context:
                regime = market_context["market_regime"]
                
                # In bearish markets, be more conservative with longs
                if regime == "bearish" and direction == "long" and confidence < 0.7:
                    params["reason"] = "Long signal not strong enough in bearish market"
                    return params
                
                # In bullish markets, be more conservative with shorts
                if regime == "bullish" and direction == "short" and confidence < 0.7:
                    params["reason"] = "Short signal not strong enough in bullish market"
                    return params
            
            # Update parameters
            params.update({
                "suitable_for_trading": True,
                "direction": direction,
                "strategy": strategy,
                "entry_type": entry_type,
                "entry_price": entry_price,
                "entry_valid_until": datetime.now() + timedelta(days=1),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": trailing_stop,
                "risk_reward": risk_reward,
                "position_size": position_size,
                "confidence": confidence
            })
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error determining trading parameters: {str(e)}")
            return {
                "suitable_for_trading": False,
                "reason": f"Error: {str(e)}"
            }
    
    def _save_opportunity(self, opportunity):
        """Save trading opportunity to database"""
        try:
            # Create document for database
            doc = {
                "symbol": opportunity["symbol"],
                "exchange": opportunity["exchange"],
                "opportunity_type": "futures",
                "expiry": opportunity["expiry"],
                "timestamp": opportunity["timestamp"],
                "data": opportunity
            }
            
            # Insert into database
            self.db.opportunity_collection.insert_one(doc)
            
        except Exception as e:
            self.logger.error(f"Error saving opportunity: {str(e)}")
    
    def execute_trade(self, symbol, exchange, expiry, direction, entry_type, entry_price, 
                     stop_loss, take_profit, position_size, strategy, order_params=None):
        """
        Execute a futures trade based on identified opportunity
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - expiry: Contract expiry date
        - direction: Trade direction ('long' or 'short')
        - entry_type: Entry order type ('market' or 'limit')
        - entry_price: Entry price (for limit orders)
        - stop_loss: Stop loss price
        - take_profit: Take profit price
        - position_size: Position size to trade (in lots)
        - strategy: Trading strategy used
        - order_params: Additional order parameters
        
        Returns:
        - Dictionary with order execution details
        """
        try:
            # Default to simulation mode
            if not order_params:
                order_params = {}
            
            simulation = order_params.get("simulation", True)
            
            if simulation:
                # Simulated trade execution
                trade_result = {
                    "status": "success",
                    "order_id": f"sim_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "symbol": symbol,
                    "exchange": exchange,
                    "expiry": expiry,
                    "direction": direction,
                    "entry_type": entry_type,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "position_size": position_size,
                    "timestamp": datetime.now(),
                    "strategy": strategy,
                    "simulated": True
                }
                
                # Log the simulated trade
                self.logger.info(f"Simulated futures trade: {trade_result}")
            else:
                # Real trading via broker integration
                try:
                    from realtime.zerodha_integration import ZerodhaTrader
                    trader = ZerodhaTrader(self.db)
                    
                    # Transform parameters for Zerodha
                    order_type = "MARKET" if entry_type == "market" else "LIMIT"
                    transaction_type = "BUY" if direction == "long" else "SELL"
                    
                    # Place the order
                    trade_result = trader.place_futures_order(
                        symbol=symbol,
                        exchange=exchange,
                        expiry=expiry,
                        transaction_type=transaction_type,
                        order_type=order_type,
                        quantity=position_size,
                        price=entry_price if order_type == "LIMIT" else None,
                        trigger_price=None
                    )
                    
                    # Set stop loss and take profit
                    if trade_result["status"] == "success":
                        trader.place_futures_sl_order(
                            symbol=symbol,
                            exchange=exchange,
                            expiry=expiry,
                            transaction_type="SELL" if transaction_type == "BUY" else "BUY",
                            quantity=position_size,
                            trigger_price=stop_loss
                        )
                        
                        # Take profit order (limit order)
                        trader.place_futures_order(
                            symbol=symbol,
                            exchange=exchange,
                            expiry=expiry,
                            transaction_type="SELL" if transaction_type == "BUY" else "BUY",
                            order_type="LIMIT",
                            quantity=position_size,
                            price=take_profit
                        )
                        
                        # Add strategy and timestamp to result
                        trade_result["strategy"] = strategy
                        trade_result["timestamp"] = datetime.now()
                        trade_result["simulated"] = False
                except Exception as broker_error:
                    self.logger.error(f"Error executing trade via broker: {str(broker_error)}")
                    return {"status": "error", "message": f"Broker execution error: {str(broker_error)}"}
            
            # Save trade to database
            self._save_trade(trade_result)
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Error executing futures trade: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _save_trade(self, trade_result):
        """Save trade execution to database"""
        try:
            # Create document for database
            doc = {
                "symbol": trade_result["symbol"],
                "exchange": trade_result["exchange"],
                "expiry": trade_result["expiry"],
                "trade_type": "futures",
                "direction": trade_result["direction"],
                "entry_time": trade_result["timestamp"],
                "entry_price": trade_result["entry_price"],
                "stop_loss": trade_result["stop_loss"],
                "take_profit": trade_result["take_profit"],
                "position_size": trade_result["position_size"],
                "order_id": trade_result["order_id"],
                "strategy": trade_result["strategy"],
                "status": "open",
                "simulated": trade_result.get("simulated", True)
            }
            
            # Insert into database
            self.db.trade_collection.insert_one(doc)
            
        except Exception as e:
            self.logger.error(f"Error saving trade: {str(e)}")
    
    def execute_calendar_spread(self, symbol, exchange, near_expiry, far_expiry, 
                              position_size, strategy, order_params=None):
        """
        Execute a futures calendar spread trade
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - near_expiry: Near month expiry date
        - far_expiry: Far month expiry date
        - position_size: Position size (in lots)
        - strategy: Trading strategy description
        - order_params: Additional order parameters
        
        Returns:
        - Dictionary with order execution details
        """
        try:
            # Default to simulation mode
            if not order_params:
                order_params = {}
            
            simulation = order_params.get("simulation", True)
            
            # Get prices for both contracts
            near_price = self._get_contract_price({
                "symbol": symbol,
                "exchange": exchange,
                "expiry": near_expiry
            })
            
            far_price = self._get_contract_price({
                "symbol": symbol,
                "exchange": exchange,
                "expiry": far_expiry
            })
            
            if not near_price or not far_price:
                return {"status": "error", "message": "Unable to get contract prices"}
            
            # Determine spread direction based on term structure
            term_structure = self._analyze_term_structure(
                symbol, exchange,
                [
                    {"symbol": symbol, "exchange": exchange, "expiry": near_expiry},
                    {"symbol": symbol, "exchange": exchange, "expiry": far_expiry}
                ]
            )
            
            # Default to contango assumption (short near, long far)
            near_direction = "short"
            far_direction = "long"
            
            if term_structure and term_structure.get("shape") == "backwardation":
                # In backwardation, reverse the directions
                near_direction = "long"
                far_direction = "short"
            
            if simulation:
                # Simulated spread execution
                spread_result = {
                    "status": "success",
                    "symbol": symbol,
                    "exchange": exchange,
                    "strategy": strategy,
                    "timestamp": datetime.now(),
                    "simulated": True,
                    "near_leg": {
                        "order_id": f"sim_near_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "expiry": near_expiry,
                        "direction": near_direction,
                        "price": near_price,
                        "position_size": position_size
                    },
                    "far_leg": {
                        "order_id": f"sim_far_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "expiry": far_expiry,
                        "direction": far_direction,
                        "price": far_price,
                        "position_size": position_size
                    },
                    "spread": {
                        "initial_value": far_price - near_price if near_direction == "short" else near_price - far_price,
                        "trade_description": f"{near_direction.capitalize()} {near_expiry.strftime('%b')} / {far_direction.capitalize()} {far_expiry.strftime('%b')}"
                    }
                }
                
                # Log the simulated spread
                self.logger.info(f"Simulated futures spread: {spread_result}")
            else:
                # Real spread via broker integration
                try:
                    from realtime.zerodha_integration import ZerodhaTrader
                    trader = ZerodhaTrader(self.db)
                    
                    # Execute near leg
                    near_result = trader.place_futures_order(
                        symbol=symbol,
                        exchange=exchange,
                        expiry=near_expiry,
                        transaction_type="BUY" if near_direction == "long" else "SELL",
                        order_type="MARKET",
                        quantity=position_size
                    )
                    
                    # Execute far leg
                    far_result = trader.place_futures_order(
                        symbol=symbol,
                        exchange=exchange,
                        expiry=far_expiry,
                        transaction_type="BUY" if far_direction == "long" else "SELL",
                        order_type="MARKET",
                        quantity=position_size
                    )
                    
                    if near_result["status"] == "success" and far_result["status"] == "success":
                        spread_result = {
                            "status": "success",
                            "symbol": symbol,
                            "exchange": exchange,
                            "strategy": strategy,
                            "timestamp": datetime.now(),
                            "simulated": False,
                            "near_leg": {
                                "order_id": near_result["order_id"],
                                "expiry": near_expiry,
                                "direction": near_direction,
                                "price": near_price,
                                "position_size": position_size
                            },
                            "far_leg": {
                                "order_id": far_result["order_id"],
                                "expiry": far_expiry,
                                "direction": far_direction,
                                "price": far_price,
                                "position_size": position_size
                            },
                            "spread": {
                                "initial_value": far_price - near_price if near_direction == "short" else near_price - far_price,
                                "trade_description": f"{near_direction.capitalize()} {near_expiry.strftime('%b')} / {far_direction.capitalize()} {far_expiry.strftime('%b')}"
                            }
                        }
                    else:
                        # Attempt to unwind any successful leg
                        if near_result["status"] == "success":
                            trader.place_futures_order(
                                symbol=symbol,
                                exchange=exchange,
                                expiry=near_expiry,
                                transaction_type="SELL" if near_direction == "long" else "BUY",
                                order_type="MARKET",
                                quantity=position_size
                            )
                        
                        if far_result["status"] == "success":
                            trader.place_futures_order(
                                symbol=symbol,
                                exchange=exchange,
                                expiry=far_expiry,
                                transaction_type="SELL" if far_direction == "long" else "BUY",
                                order_type="MARKET",
                                quantity=position_size
                            )
                        
                        return {"status": "error", "message": "Failed to execute both legs of calendar spread"}
                except Exception as broker_error:
                    self.logger.error(f"Error executing spread via broker: {str(broker_error)}")
                    return {"status": "error", "message": f"Broker execution error: {str(broker_error)}"}
            
            # Save spread trade to database
            self._save_spread_trade(spread_result)
            
            return spread_result
            
        except Exception as e:
            self.logger.error(f"Error executing calendar spread: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _save_spread_trade(self, spread_result):
        """Save spread trade execution to database"""
        try:
            # Create document for database
            doc = {
                "symbol": spread_result["symbol"],
                "exchange": spread_result["exchange"],
                "trade_type": "futures_spread",
                "timestamp": spread_result["timestamp"],
                "strategy": spread_result["strategy"],
                "near_leg": spread_result["near_leg"],
                "far_leg": spread_result["far_leg"],
                "spread": spread_result["spread"],
                "status": "open",
                "simulated": spread_result.get("simulated", True)
            }
            
            # Insert into database
            self.db.trade_collection.insert_one(doc)
            
        except Exception as e:
            self.logger.error(f"Error saving spread trade: {str(e)}")
    
    def get_futures_positions(self):
        """
        Get all current futures positions
        
        Returns:
        - List of open futures positions
        """
        try:
            # Query for open futures trades
            query = {
                "trade_type": "futures",
                "status": "open"
            }
            
            positions = list(self.db.trade_collection.find(query))
            
            # For each position, get current price and calculate unrealized P&L
            for position in positions:
                current_price = self._get_contract_price({
                    "symbol": position["symbol"],
                    "exchange": position["exchange"],
                    "expiry": position["expiry"]
                })
                
                if current_price:
                    if position["direction"] == "long":
                        unrealized_pl = (current_price - position["entry_price"]) * position["position_size"]
                        unrealized_pl_percent = ((current_price / position["entry_price"]) - 1) * 100
                    else:
                        unrealized_pl = (position["entry_price"] - current_price) * position["position_size"]
                        unrealized_pl_percent = ((position["entry_price"] / current_price) - 1) * 100
                    
                    position["current_price"] = current_price
                    position["unrealized_pl"] = unrealized_pl
                    position["unrealized_pl_percent"] = unrealized_pl_percent
                    
                    # Calculate days to expiry
                    position["days_to_expiry"] = (position["expiry"] - datetime.now()).days
            
            return {
                "status": "success",
                "positions": positions,
                "count": len(positions)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting futures positions: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def rollover_position(self, trade_id, new_expiry, order_params=None):
        """
        Rollover a futures position to a new expiry
        
        Parameters:
        - trade_id: ID of the trade to rollover
        - new_expiry: New expiry date
        - order_params: Additional order parameters
        
        Returns:
        - Dictionary with rollover details
        """
        try:
            # Get the trade from database
            trade = self.db.trade_collection.find_one({"_id": trade_id})
            
            if not trade:
                return {"status": "error", "message": "Trade not found"}
            
            if trade["status"] != "open":
                return {"status": "error", "message": "Cannot rollover closed trade"}
            
            if trade["trade_type"] != "futures":
                return {"status": "error", "message": "Only futures trades can be rolled over"}
            
            # Default to simulation mode
            if not order_params:
                order_params = {}
            
            simulation = order_params.get("simulation", True)
            
            # Get current price for existing position
            current_price = self._get_contract_price({
                "symbol": trade["symbol"],
                "exchange": trade["exchange"],
                "expiry": trade["expiry"]
            })
            
            # Get price for new contract
            new_price = self._get_contract_price({
                "symbol": trade["symbol"],
                "exchange": trade["exchange"],
                "expiry": new_expiry
            })
            
            if not current_price or not new_price:
                return {"status": "error", "message": "Unable to get contract prices"}
            
            # Calculate roll cost and P&L on current position
            roll_cost = new_price - current_price
            
            if trade["direction"] == "long":
                position_pl = (current_price - trade["entry_price"]) * trade["position_size"]
            else:
                position_pl = (trade["entry_price"] - current_price) * trade["position_size"]
            
            if simulation:
                # Simulated rollover
                rollover_result = {
                    "status": "success",
                    "original_trade_id": trade_id,
                    "symbol": trade["symbol"],
                    "exchange": trade["exchange"],
                    "original_expiry": trade["expiry"],
                    "new_expiry": new_expiry,
                    "direction": trade["direction"],
                    "position_size": trade["position_size"],
                    "original_entry_price": trade["entry_price"],
                    "new_entry_price": new_price,
                    "roll_cost": roll_cost * trade["position_size"],
                    "position_pl": position_pl,
                    "timestamp": datetime.now(),
                    "simulated": True,
                    "new_trade_id": f"sim_roll_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                }
                
                # Log the simulated rollover
                self.logger.info(f"Simulated futures rollover: {rollover_result}")
            else:
                # Real rollover via broker integration
                try:
                    from realtime.zerodha_integration import ZerodhaTrader
                    trader = ZerodhaTrader(self.db)
                    
                    # Close current position
                    close_result = trader.place_futures_order(
                        symbol=trade["symbol"],
                        exchange=trade["exchange"],
                        expiry=trade["expiry"],
                        transaction_type="SELL" if trade["direction"] == "long" else "BUY",
                        order_type="MARKET",
                        quantity=trade["position_size"]
                    )
                    
                    # Open new position
                    new_result = trader.place_futures_order(
                        symbol=trade["symbol"],
                        exchange=trade["exchange"],
                        expiry=new_expiry,
                        transaction_type="BUY" if trade["direction"] == "long" else "SELL",
                        order_type="MARKET",
                        quantity=trade["position_size"]
                    )
                    
                    if close_result["status"] == "success" and new_result["status"] == "success":
                        rollover_result = {
                            "status": "success",
                            "original_trade_id": trade_id,
                            "symbol": trade["symbol"],
                            "exchange": trade["exchange"],
                            "original_expiry": trade["expiry"],
                            "new_expiry": new_expiry,
                            "direction": trade["direction"],
                            "position_size": trade["position_size"],
                            "original_entry_price": trade["entry_price"],
                            "new_entry_price": new_price,
                            "roll_cost": roll_cost * trade["position_size"],
                            "position_pl": position_pl,
                            "timestamp": datetime.now(),
                            "simulated": False,
                            "close_order_id": close_result["order_id"],
                            "new_order_id": new_result["order_id"],
                            "new_trade_id": new_result["order_id"]
                        }
                        
                        # Set stop loss for new position
                        # Calculate similar risk percentage as original
                        original_risk_percent = abs(trade["stop_loss"] - trade["entry_price"]) / trade["entry_price"]
                        
                        if trade["direction"] == "long":
                            new_stop = new_price * (1 - original_risk_percent)
                        else:
                            new_stop = new_price * (1 + original_risk_percent)
                        
                        trader.place_futures_sl_order(
                            symbol=trade["symbol"],
                            exchange=trade["exchange"],
                            expiry=new_expiry,
                            transaction_type="SELL" if trade["direction"] == "long" else "BUY",
                            quantity=trade["position_size"],
                            trigger_price=new_stop
                        )
                        
                        rollover_result["new_stop_loss"] = new_stop
                    else:
                        return {"status": "error", "message": "Failed to execute rollover orders"}
                except Exception as broker_error:
                    self.logger.error(f"Error executing rollover via broker: {str(broker_error)}")
                    return {"status": "error", "message": f"Broker execution error: {str(broker_error)}"}
            
            # Update database records
            # Close original trade
            self.db.trade_collection.update_one(
                {"_id": trade_id},
                {
                    "$set": {
                        "status": "closed",
                        "exit_time": datetime.now(),
                        "exit_price": current_price,
                        "exit_reason": "rollover",
                        "profit_loss": position_pl,
                        "profit_loss_percent": (position_pl / (trade["entry_price"] * trade["position_size"])) * 100,
                        "rollover_details": {
                            "rolled_to_expiry": new_expiry,
                            "roll_cost": roll_cost * trade["position_size"]
                        }
                    }
                }
            )
            
            # Create new trade record
            new_trade = {
                "symbol": trade["symbol"],
                "exchange": trade["exchange"],
                "expiry": new_expiry,
                "trade_type": "futures",
                "direction": trade["direction"],
                "entry_time": datetime.now(),
                "entry_price": new_price,
                "stop_loss": rollover_result.get("new_stop_loss", None),
                "take_profit": None,  # To be determined
                "position_size": trade["position_size"],
                "order_id": rollover_result.get("new_order_id", rollover_result["new_trade_id"]),
                "strategy": trade["strategy"],
                "status": "open",
                "simulated": rollover_result.get("simulated", True),
                "rollover_details": {
                    "rolled_from_trade_id": trade_id,
                    "rolled_from_expiry": trade["expiry"],
                    "roll_cost": roll_cost * trade["position_size"]
                }
            }
            
            new_trade_id = self.db.trade_collection.insert_one(new_trade).inserted_id
            
            # Update the result with the actual database ID
            rollover_result["new_trade_id"] = new_trade_id
            
            return rollover_result
            
        except Exception as e:
            self.logger.error(f"Error rolling over position: {str(e)}")
            return {"status": "error", "message": str(e)}