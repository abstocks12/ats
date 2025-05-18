# equity.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

class EquityTrader:
    def __init__(self, db_connector):
        """Initialize the equity trader"""
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        
    def analyze_trading_opportunity(self, symbol, exchange, prediction=None, timeframe="day"):
        """
        Analyze potential trading opportunities for an equity
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - prediction: Optional prediction data
        - timeframe: Timeframe for analysis
        
        Returns:
        - Dictionary with trading opportunity details
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol, exchange, timeframe)
            if not market_data or len(market_data) < 20:
                return {"status": "error", "message": "Insufficient market data"}
            
            # Get technical analysis
            technical = self._get_technical_analysis(symbol, exchange, timeframe)
            
            # Get fundamental analysis
            fundamental = self._get_fundamental_analysis(symbol, exchange)
            
            # Get market breadth and sentiment
            market_context = self._get_market_context(symbol, exchange)
            
            # Analyze liquidity
            liquidity = self._analyze_liquidity(symbol, exchange, market_data)
            
            # Analyze volatility
            volatility = self._analyze_volatility(symbol, exchange, market_data)
            
            # Determine trading parameters
            trade_params = self._determine_trading_parameters(
                symbol, exchange, market_data, technical, 
                fundamental, market_context, prediction
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
                "fundamental_context": fundamental["summary"] if fundamental else None,
                "market_context": market_context["summary"] if market_context else None,
                "liquidity": liquidity,
                "volatility": volatility
            }
            
            # Save opportunity to database
            self._save_opportunity(opportunity)
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error analyzing equity opportunity: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_market_data(self, symbol, exchange, timeframe):
        """Get market data for analysis"""
        try:
            # Query for market data
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
            self.logger.error(f"Error getting market data: {str(e)}")
            return None
    
    def _get_technical_analysis(self, symbol, exchange, timeframe):
        """Get technical analysis for the symbol"""
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
    
    def _get_fundamental_analysis(self, symbol, exchange):
        """Get fundamental analysis for the symbol"""
        try:
            # Try to get from FundamentalAnalyzer
            try:
                from research.fundamental_analyzer import FundamentalAnalyzer
                analyzer = FundamentalAnalyzer(self.db)
                analysis = analyzer.analyze(symbol, exchange)
                return analysis
            except Exception as fund_error:
                self.logger.warning(f"Fundamental analyzer not available: {str(fund_error)}")
            
            # Fallback to database query
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "analysis_type": "fundamental"
            }
            
            analysis = self.db.analysis_collection.find_one(query)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error getting fundamental analysis: {str(e)}")
            return None
    
    def _get_market_context(self, symbol, exchange):
        """Get market breadth and sentiment analysis"""
        try:
            # Get sector for this symbol
            instrument = self.db.portfolio_collection.find_one(
                {"symbol": symbol, "exchange": exchange},
                {"sector": 1}
            )
            
            sector = instrument.get("sector") if instrument else None
            
            # Try to get market analysis
            try:
                from research.market_analysis import MarketAnalyzer
                analyzer = MarketAnalyzer(self.db)
                analysis = analyzer.get_market_context(exchange, sector)
                return analysis
            except Exception as market_error:
                self.logger.warning(f"Market analyzer not available: {str(market_error)}")
            
            # Fallback to database query
            query = {
                "exchange": exchange,
                "analysis_type": "market"
            }
            
            if sector:
                query["sector"] = sector
            
            analysis = self.db.analysis_collection.find_one(query)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error getting market context: {str(e)}")
            return None
    
    def _analyze_liquidity(self, symbol, exchange, market_data):
        """Analyze liquidity to ensure tradability"""
        try:
            if market_data is None or "volume" not in market_data:
                return {"status": "unknown"}
            
            # Calculate average daily volume
            avg_volume = market_data["volume"].mean()
            
            # Calculate average daily range
            avg_range = ((market_data["high"] - market_data["low"]) / market_data["close"]).mean() * 100
            
            # Check if liquid enough
            is_liquid = avg_volume >= 100000 and avg_range >= 1.0
            
            liquidity_rating = "High" if avg_volume >= 500000 else \
                              "Medium" if avg_volume >= 100000 else "Low"
            
            return {
                "status": "success",
                "avg_daily_volume": avg_volume,
                "avg_daily_range_percent": avg_range,
                "is_liquid": is_liquid,
                "liquidity_rating": liquidity_rating
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_volatility(self, symbol, exchange, market_data):
        """Analyze volatility for risk assessment"""
        try:
            if market_data is None:
                return {"status": "unknown"}
            
            # Calculate daily returns
            if len(market_data) > 1:
                market_data["returns"] = market_data["close"].pct_change() * 100
                
                # Calculate volatility metrics
                daily_volatility = market_data["returns"].std()
                annualized_volatility = daily_volatility * np.sqrt(252)
                
                # Calculate maximum drawdown
                rolling_max = market_data["close"].cummax()
                drawdown = ((market_data["close"] - rolling_max) / rolling_max) * 100
                max_drawdown = drawdown.min()
                
                # Classify volatility
                volatility_rating = "High" if annualized_volatility >= 30 else \
                                   "Medium" if annualized_volatility >= 15 else "Low"
                
                return {
                    "status": "success",
                    "daily_volatility": daily_volatility,
                    "annualized_volatility": annualized_volatility,
                    "max_drawdown": max_drawdown,
                    "volatility_rating": volatility_rating
                }
            
            return {"status": "unknown"}
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _determine_trading_parameters(self, symbol, exchange, market_data, technical, 
                                    fundamental, market_context, prediction):
        """Determine key trading parameters"""
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
            
            # Current market price
            current_price = market_data["close"].iloc[-1] if market_data is not None and len(market_data) > 0 else None
            if not current_price:
                params["reason"] = "Unable to determine current price"
                return params
            
            # Get direction from prediction or technical
            direction = None
            
            if prediction and "prediction" in prediction:
                direction = "long" if prediction["prediction"] == "up" else \
                           "short" if prediction["prediction"] == "down" else None
                confidence = prediction.get("confidence", 0.5)
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
                else:
                    confidence = 0.5
            else:
                params["reason"] = "No predictive signals available"
                return params
            
            if not direction:
                params["reason"] = "No clear directional bias"
                return params
            
            # Check if market context agrees with direction
            if market_context and "market_regime" in market_context:
                regime = market_context["market_regime"]
                
                # In bearish markets, be more selective with longs
                if regime == "bearish" and direction == "long" and confidence < 0.7:
                    params["reason"] = "Long signal not strong enough in bearish market"
                    return params
                
                # In bullish markets, be more selective with shorts
                if regime == "bullish" and direction == "short" and confidence < 0.7:
                    params["reason"] = "Short signal not strong enough in bullish market"
                    return params
            
            # Check if fundamental analysis contradicts direction
            if fundamental and "valuation" in fundamental:
                valuation = fundamental["valuation"]
                
                # If extremely overvalued, be more selective with longs
                if valuation == "extremely_overvalued" and direction == "long" and confidence < 0.8:
                    params["reason"] = "Long signal not strong enough for overvalued stock"
                    return params
                
                # If extremely undervalued, be more selective with shorts
                if valuation == "extremely_undervalued" and direction == "short" and confidence < 0.8:
                    params["reason"] = "Short signal not strong enough for undervalued stock"
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
                    stop_loss = entry_price * 0.95  # 5% below entry
                else:
                    stop_loss = entry_price * 1.05  # 5% above entry
            
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
            position_size = 100
            
            # Determine strategy
            strategy = self._determine_strategy(technical, fundamental, market_context, direction)
            
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
    
    def _determine_strategy(self, technical, fundamental, market_context, direction):
        """Determine the most appropriate trading strategy"""
        try:
            # Default strategy
            strategy = "momentum"
            
            # Check for mean reversion scenario
            if technical and "market_regime" in technical:
                regime = technical["market_regime"]
                
                if regime == "oversold" and direction == "long":
                    strategy = "mean_reversion_long"
                elif regime == "overbought" and direction == "short":
                    strategy = "mean_reversion_short"
            
            # Check for trend following scenario
            if technical and "trend" in technical:
                trend = technical["trend"]
                
                if trend.get("direction") == "uptrend" and trend.get("strength") in ["moderate", "strong"] and direction == "long":
                    strategy = "trend_following_long"
                elif trend.get("direction") == "downtrend" and trend.get("strength") in ["moderate", "strong"] and direction == "short":
                    strategy = "trend_following_short"
            
            # Check for breakout scenario
            if technical and "breakout" in technical:
                breakout = technical["breakout"]
                
                if breakout.get("detected") and breakout.get("direction") == "up" and direction == "long":
                    strategy = "breakout_long"
                elif breakout.get("detected") and breakout.get("direction") == "down" and direction == "short":
                    strategy = "breakout_short"
            
            # Check for fundamental catalyst
            if fundamental and "recent_catalysts" in fundamental:
                catalysts = fundamental["recent_catalysts"]
                
                if catalysts and len(catalysts) > 0:
                    if direction == "long":
                        strategy = "catalyst_long"
                    else:
                        strategy = "catalyst_short"
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error determining strategy: {str(e)}")
            return "momentum"  # Default fallback
    
    def _save_opportunity(self, opportunity):
        """Save trading opportunity to database"""
        try:
            # Create document for database
            doc = {
                "symbol": opportunity["symbol"],
                "exchange": opportunity["exchange"],
                "opportunity_type": "equity",
                "timestamp": datetime.now(),
                "data": opportunity
            }
            
            # Insert into database
            self.db.opportunity_collection.insert_one(doc)
            
        except Exception as e:
            self.logger.error(f"Error saving opportunity: {str(e)}")
    
    def execute_trade(self, symbol, exchange, direction, entry_type, entry_price, 
                     stop_loss, take_profit, position_size, strategy, order_params=None):
        """
        Execute a trade based on identified opportunity
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - direction: Trade direction ('long' or 'short')
        - entry_type: Entry order type ('market' or 'limit')
        - entry_price: Entry price (for limit orders)
        - stop_loss: Stop loss price
        - take_profit: Take profit price
        - position_size: Position size to trade
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
                self.logger.info(f"Simulated equity trade: {trade_result}")
            else:
                # Real trading via broker integration
                try:
                    from realtime.zerodha_integration import ZerodhaTrader
                    trader = ZerodhaTrader(self.db)
                    
                    # Transform parameters for Zerodha
                    order_type = "MARKET" if entry_type == "market" else "LIMIT"
                    transaction_type = "BUY" if direction == "long" else "SELL"
                    
                    # Place the order
                    trade_result = trader.place_equity_order(
                        symbol=symbol,
                        exchange=exchange,
                        transaction_type=transaction_type,
                        order_type=order_type,
                        quantity=position_size,
                        price=entry_price if order_type == "LIMIT" else None,
                        trigger_price=None
                    )
                    
                    # Set stop loss and take profit
                    if trade_result["status"] == "success":
                        trader.place_equity_sl_order(
                            symbol=symbol,
                            exchange=exchange,
                            transaction_type="SELL" if transaction_type == "BUY" else "BUY",
                            quantity=position_size,
                            trigger_price=stop_loss
                        )
                        
                        # Take profit order (limit order)
                        trader.place_equity_order(
                            symbol=symbol,
                            exchange=exchange,
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
            self.logger.error(f"Error executing equity trade: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _save_trade(self, trade_result):
        """Save trade execution to database"""
        try:
            # Create document for database
            doc = {
                "symbol": trade_result["symbol"],
                "exchange": trade_result["exchange"],
                "trade_type": "equity",
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
    
    def modify_stop_loss(self, trade_id, new_stop_loss, simulation=True):
        """
        Modify stop loss for an existing equity trade
        
        Parameters:
        - trade_id: Trade ID to modify
        - new_stop_loss: New stop loss price
        - simulation: Whether this is a simulated trade
        
        Returns:
        - Dictionary with modification result
        """
        try:
            # Get the trade from database
            trade = self.db.trade_collection.find_one({"_id": trade_id})
            
            if not trade:
                return {"status": "error", "message": "Trade not found"}
            
            if trade["status"] != "open":
                return {"status": "error", "message": "Cannot modify closed trade"}
            
            if simulation or trade.get("simulated", True):
                # Simulated modification
                self.db.trade_collection.update_one(
                    {"_id": trade_id},
                    {"$set": {"stop_loss": new_stop_loss}}
                )
                
                return {
                    "status": "success",
                    "trade_id": trade_id,
                    "new_stop_loss": new_stop_loss,
                    "simulated": True
                }
            else:
                # Real modification via broker integration
                try:
                    from realtime.zerodha_integration import ZerodhaTrader
                    trader = ZerodhaTrader(self.db)
                    
                    # First cancel existing SL order if any
                    if "sl_order_id" in trade:
                        trader.cancel_order(trade["sl_order_id"])
                    
                    # Place new SL order
                    result = trader.place_equity_sl_order(
                        symbol=trade["symbol"],
                        exchange=trade["exchange"],
                        transaction_type="SELL" if trade["direction"] == "long" else "BUY",
                        quantity=trade["position_size"],
                        trigger_price=new_stop_loss
                    )
                    
                    if result["status"] == "success":
                        # Update in database
                        self.db.trade_collection.update_one(
                            {"_id": trade_id},
                            {
                                "$set": {
                                    "stop_loss": new_stop_loss,
                                    "sl_order_id": result["order_id"]
                                }
                            }
                        )
                        
                        return {
                            "status": "success",
                            "trade_id": trade_id,
                            "new_stop_loss": new_stop_loss,
                            "sl_order_id": result["order_id"],
                            "simulated": False
                        }
                    else:
                        return {"status": "error", "message": "Failed to modify stop loss via broker"}
                        
                except Exception as broker_error:
                    self.logger.error(f"Error modifying stop loss via broker: {str(broker_error)}")
                    return {"status": "error", "message": f"Broker modification error: {str(broker_error)}"}
            
        except Exception as e:
            self.logger.error(f"Error modifying stop loss: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def close_trade(self, trade_id, close_price=None, close_reason="manual", simulation=True):
        """
        Close an existing equity trade
        
        Parameters:
        - trade_id: Trade ID to close
        - close_price: Optional closing price (for simulation)
        - close_reason: Reason for closing the trade
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
            
            # For simulation, need to provide a close price if not provided
            if simulation or trade.get("simulated", True):
                if not close_price:
                    # Get current market price
                    current_price = self._get_current_price(trade["symbol"], trade["exchange"])
                    if not current_price:
                        return {"status": "error", "message": "Unable to determine current price"}
                    
                    close_price = current_price
                
                # Calculate P&L
                if trade["direction"] == "long":
                    profit_loss = (close_price - trade["entry_price"]) * trade["position_size"]
                    profit_loss_percent = ((close_price / trade["entry_price"]) - 1) * 100
                else:
                    profit_loss = (trade["entry_price"] - close_price) * trade["position_size"]
                    profit_loss_percent = ((trade["entry_price"] / close_price) - 1) * 100
                
                # Update in database
                self.db.trade_collection.update_one(
                    {"_id": trade_id},
                    {
                        "$set": {
                            "status": "closed",
                            "exit_time": datetime.now(),
                            "exit_price": close_price,
                            "exit_reason": close_reason,
                            "profit_loss": profit_loss,
                            "profit_loss_percent": profit_loss_percent
                        }
                    }
                )
                
                return {
                    "status": "success",
                    "trade_id": trade_id,
                    "exit_price": close_price,
                    "exit_reason": close_reason,
                    "profit_loss": profit_loss,
                    "profit_loss_percent": profit_loss_percent,
                    "simulated": True
                }
            else:
                # Real closure via broker integration
                try:
                    from realtime.zerodha_integration import ZerodhaTrader
                    trader = ZerodhaTrader(self.db)
                    
                    # Cancel any pending orders first
                    if "sl_order_id" in trade:
                        trader.cancel_order(trade["sl_order_id"])
                    if "tp_order_id" in trade:
                        trader.cancel_order(trade["tp_order_id"])
                    
                    # Place closing order
                    result = trader.place_equity_order(
                        symbol=trade["symbol"],
                        exchange=trade["exchange"],
                        transaction_type="SELL" if trade["direction"] == "long" else "BUY",
                        order_type="MARKET",
                        quantity=trade["position_size"]
                    )
                    
                    if result["status"] == "success":
                        # Need to wait for order execution to get actual price
                        # For now, use current market price
                        current_price = trader.get_ltp(trade["symbol"], trade["exchange"])
                        
                        # Calculate P&L
                        if trade["direction"] == "long":
                            profit_loss = (current_price - trade["entry_price"]) * trade["position_size"]
                            profit_loss_percent = ((current_price / trade["entry_price"]) - 1) * 100
                        else:
                            profit_loss = (trade["entry_price"] - current_price) * trade["position_size"]
                            profit_loss_percent = ((trade["entry_price"] / current_price) - 1) * 100
                        
                        # Update in database
                        self.db.trade_collection.update_one(
                            {"_id": trade_id},
                            {
                                "$set": {
                                    "status": "closed",
                                    "exit_time": datetime.now(),
                                    "exit_price": current_price,
                                    "exit_reason": close_reason,
                                    "profit_loss": profit_loss,
                                    "profit_loss_percent": profit_loss_percent,
                                    "closing_order_id": result["order_id"]
                                }
                            }
                        )
                        
                        return {
                            "status": "success",
                            "trade_id": trade_id,
                            "exit_price": current_price,
                            "exit_reason": close_reason,
                            "profit_loss": profit_loss,
                            "profit_loss_percent": profit_loss_percent,
                            "closing_order_id": result["order_id"],
                            "simulated": False
                        }
                    else:
                        return {"status": "error", "message": "Failed to close trade via broker"}
                        
                except Exception as broker_error:
                    self.logger.error(f"Error closing trade via broker: {str(broker_error)}")
                    return {"status": "error", "message": f"Broker closure error: {str(broker_error)}"}
            
        except Exception as e:
            self.logger.error(f"Error closing trade: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_equity_positions(self):
        """
        Get all current equity positions
        
        Returns:
        - List of open equity positions
        """
        try:
            # Query for open equity trades
            query = {
                "trade_type": "equity",
                "status": "open"
            }
            
            positions = list(self.db.trade_collection.find(query))
            
            # For each position, get current price and calculate unrealized P&L
            for position in positions:
                current_price = self._get_current_price(position["symbol"], position["exchange"])
                
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
            
            return {
                "status": "success",
                "positions": positions,
                "count": len(positions)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting equity positions: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_current_price(self, symbol, exchange):
        """Get current price for a symbol"""
        try:
            # Try real-time price first
            try:
                from realtime.zerodha_integration import ZerodhaTrader
                trader = ZerodhaTrader(self.db)
                price = trader.get_ltp(symbol, exchange)
                if price:
                    return price
            except Exception as broker_error:
                self.logger.warning(f"Unable to get real-time price: {str(broker_error)}")
            
            # Fallback to latest price from database
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": "equity"
            }
            
            latest_data = self.db.market_data_collection.find_one(
                query,
                {"close": 1}
            )
            
            if latest_data and "close" in latest_data:
                return latest_data["close"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            return None
    
    def check_corporate_actions(self, symbol, exchange):
        """
        Check for upcoming corporate actions that might affect trading
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        
        Returns:
        - Dictionary with corporate action details
        """
        try:
            # Query for corporate actions
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "data_type": "corporate_action"
            }
            
            # Get upcoming actions (future dates)
            corporate_actions = list(self.db.market_metadata_collection.find(
                query
            ).sort("date", 1))
            
            upcoming_actions = []
            for action in corporate_actions:
                if action["date"] > datetime.now():
                    upcoming_actions.append(action)
            
            # Check for immediate concerns (actions in next 7 days)
            concerns = []
            for action in upcoming_actions:
                days_to_action = (action["date"] - datetime.now()).days
                
                if days_to_action <= 7:
                    concerns.append({
                        "type": action["action_type"],
                        "date": action["date"],
                        "days_to_action": days_to_action,
                        "details": action.get("details", "")
                    })
            
            return {
                "status": "success",
                "upcoming_actions": upcoming_actions,
                "concerns": concerns,
                "has_immediate_concerns": len(concerns) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error checking corporate actions: {str(e)}")
            return {"status": "error", "message": str(e)}