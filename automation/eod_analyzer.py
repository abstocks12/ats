# automation/eod_analyzer.py
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class EODAnalyzer:
    """
    End-of-day analysis system.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the end-of-day analyzer.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("EOD analyzer initialized")
        
    def analyze(self):
        """
        Perform end-of-day analysis.
        
        Returns:
            dict: Analysis results
        """
        try:
            self.logger.info("Starting end-of-day analysis")
            
            # Analyze market performance
            market_analysis = self._analyze_market_performance()
            
            # Analyze trading performance
            trading_analysis = self._analyze_trading_performance()
            
            # Analyze sector performance
            sector_analysis = self._analyze_sector_performance()
            
            # Analyze prediction accuracy
            prediction_analysis = self._analyze_prediction_accuracy()
            
            # Identify patterns
            pattern_analysis = self._identify_patterns()
            
            # Generate key observations
            observations = self._generate_observations(
                market_analysis, 
                trading_analysis, 
                sector_analysis, 
                prediction_analysis, 
                pattern_analysis
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                market_analysis,
                trading_analysis,
                sector_analysis,
                prediction_analysis,
                pattern_analysis,
                observations
            )
            
            # Compile results
            results = {
                "date": datetime.now().date(),
                "market_analysis": market_analysis,
                "trading_analysis": trading_analysis,
                "sector_analysis": sector_analysis,
                "prediction_analysis": prediction_analysis,
                "pattern_analysis": pattern_analysis,
                "observations": observations,
                "recommendations": recommendations
            }
            
            # Store results in database
            if self.db:
                self.db.eod_analysis.insert_one(results)
                
            self.logger.info("End-of-day analysis completed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in EOD analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_market_performance(self):
        """
        Analyze market performance for the day.
        
        Returns:
            dict: Market performance analysis
        """
        try:
            self.logger.info("Analyzing market performance")
            
            # Get market data for key indices
            indices = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
            exchange = "NSE"
            
            indices_data = {}
            
            # Get today's data for each index
            for index in indices:
                try:
                    # Get data from database
                    data = self._get_daily_data(index, exchange)
                    
                    if data:
                        # Calculate key metrics
                        open_price = data.get('open', 0)
                        high_price = data.get('high', 0)
                        low_price = data.get('low', 0)
                        close_price = data.get('close', 0)
                        prev_close = data.get('prev_close', 0)
                        
                        # Calculate performance metrics
                        if prev_close > 0:
                            daily_change = (close_price - prev_close) / prev_close * 100
                        else:
                            daily_change = 0
                            
                        daily_range = (high_price - low_price) / open_price * 100 if open_price > 0 else 0
                        
                        # Calculate intraday patterns
                        if open_price > 0 and close_price > 0:
                            open_to_close = (close_price - open_price) / open_price * 100
                            if open_price < close_price:
                                if low_price == open_price:
                                    pattern = "bullish_trend"
                                else:
                                    pattern = "bullish_reversal"
                            else:
                                if high_price == open_price:
                                    pattern = "bearish_trend"
                                else:
                                    pattern = "bearish_reversal"
                        else:
                            open_to_close = 0
                            pattern = "unknown"
                        
                        # Store results
                        indices_data[index] = {
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "prev_close": prev_close,
                            "daily_change": daily_change,
                            "daily_range": daily_range,
                            "open_to_close": open_to_close,
                            "pattern": pattern
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing {index}: {e}")
            
            # Analyze overall market
            market_analysis = {
                "indices": indices_data,
                "overall_sentiment": self._determine_market_sentiment(indices_data),
                "volatility": self._calculate_market_volatility(indices_data),
                "breadth": self._analyze_market_breadth(),
                "sector_rotation": self._analyze_sector_rotation()
            }
            
            return market_analysis
            
        except Exception as e:
            self.logger.error(f"Error in market performance analysis: {e}")
            return {"error": str(e)}
            
    def _analyze_trading_performance(self):
        """
        Analyze trading performance for the day.
        
        Returns:
            dict: Trading performance analysis
        """
        try:
            self.logger.info("Analyzing trading performance")
            
            # Get position manager
            from trading.position_manager import PositionManager
            position_manager = PositionManager(self.db)
            
            # Get today's trades
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            trades = list(self.db.trades.find({
                "$or": [
                    {"entry_time": {"$gte": today, "$lt": tomorrow}},
                    {"exit_time": {"$gte": today, "$lt": tomorrow}}
                ]
            }))
            
            # Analyze trades
            if not trades:
                return {
                    "trades_executed": 0,
                    "message": "No trades executed today"
                }
                
            # Calculate trade metrics
            num_trades = len(trades)
            winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            if num_trades > 0:
                win_rate = win_count / num_trades * 100
            else:
                win_rate = 0
                
            # Calculate profit metrics
            total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
            total_loss = sum(t.get('profit_loss', 0) for t in losing_trades)
            net_pnl = total_profit + total_loss
            
            # Calculate average metrics
            avg_win = total_profit / win_count if win_count > 0 else 0
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            
            # Calculate profit factor
            if total_loss != 0:
                profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
            else:
                profit_factor = float('inf') if total_profit > 0 else 0
                
            # Get daily performance
            daily_performance = position_manager.get_daily_performance()
            
            # Analyze by strategy
            strategy_performance = {}
            
            for trade in trades:
                strategy = trade.get('strategy', 'unknown')
                
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        "trades": 0,
                        "winning_trades": 0,
                        "profit_loss": 0
                    }
                    
                strategy_performance[strategy]["trades"] += 1
                
                if trade.get('profit_loss', 0) > 0:
                    strategy_performance[strategy]["winning_trades"] += 1
                    
                strategy_performance[strategy]["profit_loss"] += trade.get('profit_loss', 0)
                
            # Calculate win rate and average return per strategy
            for strategy in strategy_performance:
                if strategy_performance[strategy]["trades"] > 0:
                    strategy_performance[strategy]["win_rate"] = (
                        strategy_performance[strategy]["winning_trades"] / 
                        strategy_performance[strategy]["trades"] * 100
                    )
                    
                    strategy_performance[strategy]["avg_return"] = (
                        strategy_performance[strategy]["profit_loss"] / 
                        strategy_performance[strategy]["trades"]
                    )
                else:
                    strategy_performance[strategy]["win_rate"] = 0
                    strategy_performance[strategy]["avg_return"] = 0
            
            # Create trading analysis
            trading_analysis = {
                "trades_executed": num_trades,
                "winning_trades": win_count,
                "losing_trades": loss_count,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "net_pnl": net_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "daily_pnl": daily_performance.get('daily_pnl', 0),
                "daily_pnl_percent": daily_performance.get('daily_pnl_percent', 0),
                "strategy_performance": strategy_performance
            }
            
            return trading_analysis
            
        except Exception as e:
            self.logger.error(f"Error in trading performance analysis: {e}")
            return {"error": str(e)}
            
    def _analyze_sector_performance(self):
        """
        Analyze sector performance for the day.
        
        Returns:
            dict: Sector performance analysis
        """
        try:
            self.logger.info("Analyzing sector performance")
            
            # Get sector data
            sectors = [
                "NIFTY AUTO", "NIFTY BANK", "NIFTY FMCG", "NIFTY IT", 
                "NIFTY METAL", "NIFTY PHARMA", "NIFTY REALTY", "NIFTY ENERGY"
            ]
            exchange = "NSE"
            
            sector_data = {}
            
            # Get today's data for each sector
            for sector in sectors:
                try:
                    # Get data from database
                    data = self._get_daily_data(sector, exchange)
                    
                    if data:
                        # Calculate performance metrics
                        if data.get('prev_close', 0) > 0:
                            daily_change = (data.get('close', 0) - data.get('prev_close', 0)) / data.get('prev_close', 0) * 100
                        else:
                            daily_change = 0
                            
                        # Store results
                        sector_name = sector.replace("NIFTY ", "")
                        sector_data[sector_name] = {
                            "close": data.get('close', 0),
                            "daily_change": daily_change
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing sector {sector}: {e}")
            
            # Sort sectors by performance
            if sector_data:
                top_sectors = sorted(
                    sector_data.items(), 
                    key=lambda x: x[1]['daily_change'], 
                    reverse=True
                )
                
                bottom_sectors = sorted(
                    sector_data.items(), 
                    key=lambda x: x[1]['daily_change']
                )
                
                # Get top and bottom 3
                top_3 = top_sectors[:3]
                bottom_3 = bottom_sectors[:3]
            else:
                top_3 = []
                bottom_3 = []
            
            # Analyze sector rotation
            rotation_analysis = {
                "outperforming_sectors": [s[0] for s in top_3],
                "underperforming_sectors": [s[0] for s in bottom_3],
                "rotation_pattern": self._identify_rotation_pattern(sector_data),
                "relative_strength": self._calculate_relative_strength(sector_data)
            }
            
            # Compile sector analysis
            sector_analysis = {
                "sectors": sector_data,
                "top_sectors": [{"name": s[0], "change": s[1]['daily_change']} for s in top_3],
                "bottom_sectors": [{"name": s[0], "change": s[1]['daily_change']} for s in bottom_3],
                "rotation_analysis": rotation_analysis
            }
            
            return sector_analysis
            
        except Exception as e:
            self.logger.error(f"Error in sector performance analysis: {e}")
            return {"error": str(e)}
            
    def _analyze_prediction_accuracy(self):
        """
        Analyze prediction accuracy for the day.
        
        Returns:
            dict: Prediction accuracy analysis
        """
        try:
            self.logger.info("Analyzing prediction accuracy")
            
            # Get yesterday's predictions
            yesterday = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            predictions = list(self.db.predictions.find({
                "date": {"$gte": yesterday, "$lt": today},
                "for_date": {"$gte": today, "$lt": today + timedelta(days=1)}
            }))
            
            if not predictions:
                return {
                    "predictions": 0,
                    "message": "No predictions found for today"
                }
                
            # Calculate accuracy metrics
            num_predictions = len(predictions)
            correct_predictions = 0
            confidence_sum = 0
            
            # Analyze each prediction
            for prediction in predictions:
                symbol = prediction.get('symbol')
                exchange = prediction.get('exchange')
                predicted_direction = prediction.get('prediction')
                confidence = prediction.get('confidence', 0)
                
                confidence_sum += confidence
                
                # Get actual data
                actual_data = self._get_daily_data(symbol, exchange)
                
                if actual_data:
                    # Determine actual direction
                    if actual_data.get('close', 0) > actual_data.get('prev_close', 0):
                        actual_direction = "up"
                    else:
                        actual_direction = "down"
                        
                    # Check if prediction was correct
                    if predicted_direction == actual_direction:
                        correct_predictions += 1
                        
            # Calculate accuracy metrics
            accuracy = (correct_predictions / num_predictions * 100) if num_predictions > 0 else 0
            avg_confidence = (confidence_sum / num_predictions) if num_predictions > 0 else 0
            
            # Analyze by confidence level
            high_confidence_predictions = [p for p in predictions if p.get('confidence', 0) >= 0.7]
            medium_confidence_predictions = [p for p in predictions if 0.5 <= p.get('confidence', 0) < 0.7]
            low_confidence_predictions = [p for p in predictions if p.get('confidence', 0) < 0.5]
            
            # Calculate accuracy by confidence level
            high_correct = sum(1 for p in high_confidence_predictions if self._is_prediction_correct(p))
            medium_correct = sum(1 for p in medium_confidence_predictions if self._is_prediction_correct(p))
            low_correct = sum(1 for p in low_confidence_predictions if self._is_prediction_correct(p))
            
            high_accuracy = (high_correct / len(high_confidence_predictions) * 100) if high_confidence_predictions else 0
            medium_accuracy = (medium_correct / len(medium_confidence_predictions) * 100) if medium_confidence_predictions else 0
            low_accuracy = (low_correct / len(low_confidence_predictions) * 100) if low_confidence_predictions else 0
            
            # Create prediction analysis
            prediction_analysis = {
                "predictions": num_predictions,
                "correct_predictions": correct_predictions,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "confidence_levels": {
                    "high": {
                        "count": len(high_confidence_predictions),
                        "correct": high_correct,
                        "accuracy": high_accuracy
                    },
                    "medium": {
                        "count": len(medium_confidence_predictions),
                        "correct": medium_correct,
                        "accuracy": medium_accuracy
                    },
                    "low": {
                        "count": len(low_confidence_predictions),
                        "correct": low_correct,
                        "accuracy": low_accuracy
                    }
                }
            }
            
            return prediction_analysis
            
        except Exception as e:
            self.logger.error(f"Error in prediction accuracy analysis: {e}")
            return {"error": str(e)}
            
    def _identify_patterns(self):
        """
        Identify market patterns from the day's data.
        
        Returns:
            dict: Pattern analysis results
        """
        try:
            self.logger.info("Identifying market patterns")
            
            # Get NIFTY data for the past week
            nifty_data = self._get_nifty_data(days=5)
            
            # Identify price patterns
            price_patterns = self._identify_price_patterns(nifty_data)
            
            # Identify volume patterns
            volume_patterns = self._identify_volume_patterns(nifty_data)
            
            # Identify technical patterns
            technical_patterns = self._identify_technical_patterns(nifty_data)
            
            # Create pattern analysis
            pattern_analysis = {
                "price_patterns": price_patterns,
                "volume_patterns": volume_patterns,
                "technical_patterns": technical_patterns,
                "overall_pattern": self._determine_overall_pattern(
                    price_patterns, 
                    volume_patterns, 
                    technical_patterns
                )
            }
            
            return pattern_analysis
            
        except Exception as e:
            self.logger.error(f"Error in pattern identification: {e}")
            return {"error": str(e)}
            
    def _generate_observations(self, market_analysis, trading_analysis, 
                             sector_analysis, prediction_analysis, pattern_analysis):
        """
        Generate key observations from analysis results.
        
        Args:
            market_analysis (dict): Market performance analysis
            trading_analysis (dict): Trading performance analysis
            sector_analysis (dict): Sector performance analysis
            prediction_analysis (dict): Prediction accuracy analysis
            pattern_analysis (dict): Pattern analysis results
            
        Returns:
            list: Key observations
        """
        try:
            self.logger.info("Generating observations")
            
            observations = []
            
            # Market observations
            if "indices" in market_analysis:
                nifty_data = market_analysis["indices"].get("NIFTY", {})
                
                if nifty_data:
                    daily_change = nifty_data.get("daily_change", 0)
                    
                    if abs(daily_change) > 1:
                        direction = "gained" if daily_change > 0 else "lost"
                        observations.append(
                            f"NIFTY {direction} {abs(daily_change):.2f}% today, "
                            f"showing {nifty_data.get('pattern', 'neutral')} pattern"
                        )
                        
            # Market sentiment
            if "overall_sentiment" in market_analysis:
                sentiment = market_analysis["overall_sentiment"]
                observations.append(f"Overall market sentiment is {sentiment}")
                
            # Sector rotation
            if "rotation_analysis" in sector_analysis:
                rotation = sector_analysis["rotation_analysis"].get("rotation_pattern", "neutral")
                observations.append(f"Sector rotation shows {rotation} pattern")
                
                # Top sectors
                if "top_sectors" in sector_analysis and sector_analysis["top_sectors"]:
                    top_sector = sector_analysis["top_sectors"][0]
                    observations.append(
                        f"{top_sector['name']} was the top performing sector with "
                        f"{top_sector['change']:.2f}% gain"
                    )
                    
            # Trading performance
            if "net_pnl" in trading_analysis:
                net_pnl = trading_analysis["net_pnl"]
                
                if net_pnl > 0:
                    observations.append(
                        f"Trading was profitable today with ₹{net_pnl:.2f} net profit "
                        f"({trading_analysis.get('daily_pnl_percent', 0):.2f}%)"
                    )
                elif net_pnl < 0:
                    observations.append(
                        f"Trading resulted in a loss of ₹{abs(net_pnl):.2f} "
                        f"({abs(trading_analysis.get('daily_pnl_percent', 0)):.2f}%)"
                    )
                    
                # Win rate
                win_rate = trading_analysis.get("win_rate", 0)
                observations.append(
                    f"Win rate was {win_rate:.1f}% with "
                    f"{trading_analysis.get('winning_trades', 0)} winning trades and "
                    f"{trading_analysis.get('losing_trades', 0)} losing trades"
                )
                
            # Prediction accuracy
            if "accuracy" in prediction_analysis:
                accuracy = prediction_analysis["accuracy"]
                
                observations.append(
                    f"Prediction accuracy was {accuracy:.1f}% across "
                    f"{prediction_analysis.get('predictions', 0)} predictions"
                )
                
                # Confidence levels
                high_accuracy = prediction_analysis.get("confidence_levels", {}).get("high", {}).get("accuracy", 0)
                
                if high_accuracy > 0:
                    observations.append(
                        f"High confidence predictions had {high_accuracy:.1f}% accuracy"
                    )
                    
            # Pattern analysis
            if "overall_pattern" in pattern_analysis:
                pattern = pattern_analysis["overall_pattern"]
                observations.append(f"Market shows {pattern} pattern")
                
            return observations
            
        except Exception as e:
            self.logger.error(f"Error generating observations: {e}")
            return ["Error generating observations"]
            
    def _generate_recommendations(self, market_analysis, trading_analysis, 
                                sector_analysis, prediction_analysis, pattern_analysis,
                                observations):
        """
        Generate recommendations based on analysis results.
        
        Args:
            market_analysis (dict): Market performance analysis
            trading_analysis (dict): Trading performance analysis
            sector_analysis (dict): Sector performance analysis
            prediction_analysis (dict): Prediction accuracy analysis
            pattern_analysis (dict): Pattern analysis results
            observations (list): Key observations
            
        Returns:
            list: Recommendations
        """
        try:
            self.logger.info("Generating recommendations")
            
            recommendations = []
            
            # Market recommendations
            sentiment = market_analysis.get("overall_sentiment", "neutral")
            volatility = market_analysis.get("volatility", {}).get("level", "normal")
            
            if sentiment == "bullish":
                if volatility == "high":
                    recommendations.append(
                        "Consider bullish positions with strict risk management due to high volatility"
                    )
                else:
                    recommendations.append(
                        "Focus on trend-following strategies in the bullish environment"
                    )
            elif sentiment == "bearish":
                if volatility == "high":
                    recommendations.append(
                        "Consider hedging positions and reducing exposure in this bearish, volatile market"
                    )
                else:
                    recommendations.append(
                        "Look for short opportunities while maintaining disciplined stop management"
                    )
            else:  # neutral
                recommendations.append(
                    "Focus on range-bound strategies and wait for clearer directional signals"
                )
                
            # Sector recommendations
            if "top_sectors" in sector_analysis and sector_analysis["top_sectors"]:
                top_sectors = [s["name"] for s in sector_analysis["top_sectors"]]
                
                recommendations.append(
                    f"Consider opportunities in outperforming sectors: {', '.join(top_sectors)}"
                )
                
            # Strategy recommendations
            if "strategy_performance" in trading_analysis:
                strategy_perf = trading_analysis["strategy_performance"]
                
                # Find best performing strategy
                best_strategy = None
                best_pnl = float('-inf')
                
                for strategy, data in strategy_perf.items():
                    if data.get("profit_loss", 0) > best_pnl and data.get("trades", 0) >= 3:
                        best_pnl = data.get("profit_loss", 0)
                        best_strategy = strategy
                        
                if best_strategy and best_pnl > 0:
                    recommendations.append(
                        f"Prioritize {best_strategy} strategy which performed well today"
                    )
                    
                # Find worst performing strategy
                worst_strategy = None
                worst_pnl = float('inf')
                
                for strategy, data in strategy_perf.items():
                    if data.get("profit_loss", 0) < worst_pnl and data.get("trades", 0) >= 3:
                        worst_pnl = data.get("profit_loss", 0)
                        worst_strategy = strategy
                        
                if worst_strategy and worst_pnl < 0:
                    recommendations.append(
                        f"Review and potentially adjust parameters for {worst_strategy} strategy"
                    )
                    
            # Prediction recommendations
            if "confidence_levels" in prediction_analysis:
                confidence_levels = prediction_analysis["confidence_levels"]
                
                high_accuracy = confidence_levels.get("high", {}).get("accuracy", 0)
                medium_accuracy = confidence_levels.get("medium", {}).get("accuracy", 0)
                low_accuracy = confidence_levels.get("low", {}).get("accuracy", 0)
                
                if high_accuracy > 65:
                    recommendations.append(
                        "Continue to prioritize high confidence predictions (>70%) which showed good accuracy"
                    )
                elif medium_accuracy > high_accuracy:
                    recommendations.append(
                        "Consider medium confidence predictions which outperformed high confidence ones today"
                    )
                    
                if low_accuracy < 40 and confidence_levels.get("low", {}).get("count", 0) > 5:
                    recommendations.append(
                        "Avoid low confidence predictions which showed poor accuracy"
                    )
                    
            # Pattern recommendations
            overall_pattern = pattern_analysis.get("overall_pattern", "neutral")
            
            if "reversal" in overall_pattern:
                recommendations.append(
                    "Be alert for potential trend reversal; consider adjusting stop losses"
                )
            elif "continuation" in overall_pattern:
                recommendations.append(
                    "Look for trend continuation setups in the current market direction"
                )
            elif "consolidation" in overall_pattern:
                recommendations.append(
                    "Prepare for breakout opportunities after this consolidation phase"
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
            
    def _get_daily_data(self, symbol, exchange):
        """
        Get daily market data for a symbol.
        
        Args:
            symbol (str): Symbol name
            exchange (str): Exchange name
            
        Returns:
            dict: Daily market data
        """
        try:
            if not self.db:
                return None
                
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            # Get today's market data
            data = self.db.market_data.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "day",
                "timestamp": {"$gte": today, "$lt": tomorrow}
            })
            
            if not data:
                return None
                
            # Get previous close
            yesterday = today - timedelta(days=1)
            
            prev_data = self.db.market_data.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "day",
                "timestamp": {"$gte": yesterday, "$lt": today}
            })
            
            prev_close = prev_data.get('close', 0) if prev_data else 0
            
            # Create result
            result = {
                "open": data.get('open', 0),
                "high": data.get('high', 0),
                "low": data.get('low', 0),
                "close": data.get('close', 0),
                "volume": data.get('volume', 0),
                "prev_close": prev_close
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting daily data for {symbol}: {e}")
            return None
            
    def _get_nifty_data(self, days=5):
        """
        Get NIFTY data for the specified number of days.
        
        Args:
            days (int): Number of days
            
        Returns:
            list: List of daily data dictionaries
        """
        try:
            if not self.db:
                return []
                
            # Get end date (today)
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get start date
            start_date = end_date - timedelta(days=days)
            
            # Get market data
            cursor = self.db.market_data.find({
                "symbol": "NIFTY",
                "exchange": "NSE",
                "timeframe": "day",
                "timestamp": {"$gte": start_date, "$lt": end_date + timedelta(days=1)}
            }).sort("timestamp", 1)
            
            # Convert to list
            data = list(cursor)
            
            # Format data
            formatted_data = []
            
            for i, item in enumerate(data):
                # Get previous close
                prev_close = data[i-1].get('close', 0) if i > 0 else 0
                
                # Create formatted item
                formatted_item = {
                    "date": item.get('timestamp'),
                    "open": item.get('open', 0),
                    "high": item.get('high', 0),
                    "low": item.get('low', 0),
                    "close": item.get('close', 0),
                    "volume": item.get('volume', 0),
                    "prev_close": prev_close
                }
                
                formatted_data.append(formatted_item)
                
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error getting NIFTY data: {e}")
            return []
            
    def _determine_market_sentiment(self, indices_data):
        """
        Determine overall market sentiment.
        
        Args:
            indices_data (dict): Index performance data
            
        Returns:
            str: Market sentiment (bullish, bearish, neutral)
        """
        try:
            # Count positive and negative indices
            positive_count = 0
            negative_count = 0
            
            for index, data in indices_data.items():
                if data.get('daily_change', 0) > 0:
                    positive_count += 1
                elif data.get('daily_change', 0) < 0:
                    negative_count += 1
                    
            # Check NIFTY specifically
            nifty_data = indices_data.get('NIFTY', {})
            nifty_change = nifty_data.get('daily_change', 0)
            
            # Determine sentiment
            if positive_count > negative_count:
                if nifty_change > 0.5:
                    return "bullish"
                else:
                    return "mildly bullish"
            elif negative_count > positive_count:
                if nifty_change < -0.5:
                    return "bearish"
                else:
                    return "mildly bearish"
            else:
                return "neutral"
                
        except Exception as e:
            self.logger.error(f"Error determining market sentiment: {e}")
            return "neutral"
            
    def _calculate_market_volatility(self, indices_data):
        """
        Calculate market volatility.
        
        Args:
            indices_data (dict): Index performance data
            
        Returns:
            dict: Volatility data
        """
        try:
            # Get NIFTY data
            nifty_data = indices_data.get('NIFTY', {})
            
            if not nifty_data:
                return {"level": "unknown"}
                
            # Calculate volatility metrics
            daily_range = nifty_data.get('daily_range', 0)
            
            # Get historical volatility
            avg_range = self._get_avg_daily_range("NIFTY", "NSE", days=20)
            
            # Determine volatility level
            # Determine volatility level
            if daily_range > avg_range * 1.5:
                level = "high"
            elif daily_range < avg_range * 0.5:
                level = "low"
            else:
                level = "normal"
                
            # Create volatility data
            volatility_data = {
                "daily_range": daily_range,
                "average_range": avg_range,
                "level": level,
                "relative_volatility": daily_range / avg_range if avg_range > 0 else 1.0
            }
            
            return volatility_data
            
        except Exception as e:
            self.logger.error(f"Error calculating market volatility: {e}")
            return {"level": "unknown"}
            
    def _get_avg_daily_range(self, symbol, exchange, days=20):
        """
        Get average daily range for a symbol.
        
        Args:
            symbol (str): Symbol name
            exchange (str): Exchange name
            days (int): Number of days
            
        Returns:
            float: Average daily range
        """
        try:
            if not self.db:
                return 0
                
            # Get end date (yesterday)
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get start date
            start_date = end_date - timedelta(days=days)
            
            # Get market data
            cursor = self.db.market_data.find({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "day",
                "timestamp": {"$gte": start_date, "$lt": end_date}
            })
            
            # Calculate daily ranges
            ranges = []
            
            for item in cursor:
                open_price = item.get('open', 0)
                high_price = item.get('high', 0)
                low_price = item.get('low', 0)
                
                if open_price > 0:
                    daily_range = (high_price - low_price) / open_price * 100
                    ranges.append(daily_range)
                    
            # Calculate average
            if ranges:
                return sum(ranges) / len(ranges)
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"Error getting average daily range: {e}")
            return 0
            
    def _analyze_market_breadth(self):
        """
        Analyze market breadth.
        
        Returns:
            dict: Market breadth data
        """
        try:
            if not self.db:
                return {"breadth": "unknown"}
                
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            # Get market breadth data
            breadth_data = self.db.market_breadth.find_one({
                "date": {"$gte": today, "$lt": tomorrow}
            })
            
            if not breadth_data:
                return {"breadth": "unknown"}
                
            # Calculate breadth metrics
            advances = breadth_data.get('advances', 0)
            declines = breadth_data.get('declines', 0)
            unchanged = breadth_data.get('unchanged', 0)
            
            total = advances + declines + unchanged
            
            if total > 0:
                advance_percent = advances / total * 100
                decline_percent = declines / total * 100
                
                # Calculate AD ratio
                ad_ratio = advances / declines if declines > 0 else float('inf')
                
                # Determine breadth
                if advance_percent > 65:
                    breadth = "strongly positive"
                elif advance_percent > 55:
                    breadth = "positive"
                elif advance_percent < 35:
                    breadth = "strongly negative"
                elif advance_percent < 45:
                    breadth = "negative"
                else:
                    breadth = "neutral"
            else:
                advance_percent = 0
                decline_percent = 0
                ad_ratio = 1.0
                breadth = "unknown"
                
            # Create breadth data
            breadth_result = {
                "advances": advances,
                "declines": declines,
                "unchanged": unchanged,
                "advance_percent": advance_percent,
                "decline_percent": decline_percent,
                "ad_ratio": ad_ratio,
                "breadth": breadth
            }
            
            return breadth_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing market breadth: {e}")
            return {"breadth": "unknown"}
            
    def _analyze_sector_rotation(self):
        """
        Analyze sector rotation patterns.
        
        Returns:
            dict: Sector rotation data
        """
        try:
            # Import sector rotation analyzer
            from ml.prediction.sector_rotation_analyzer import SectorRotationAnalyzer
            
            # Initialize analyzer
            if self.db:
                analyzer = SectorRotationAnalyzer(self.db)
                
                # Analyze rotation
                rotation_data = analyzer.analyze_daily_rotation()
                
                return rotation_data
            else:
                return {"rotation": "unknown"}
                
        except Exception as e:
            self.logger.error(f"Error analyzing sector rotation: {e}")
            return {"rotation": "unknown"}
            
    def _identify_rotation_pattern(self, sector_data):
        """
        Identify sector rotation pattern.
        
        Args:
            sector_data (dict): Sector performance data
            
        Returns:
            str: Rotation pattern
        """
        try:
            if not sector_data:
                return "unknown"
                
            # Check cyclical vs defensive
            cyclical_sectors = ["AUTO", "BANK", "METAL", "REALTY"]
            defensive_sectors = ["FMCG", "PHARMA"]
            technology_sectors = ["IT"]
            energy_sectors = ["ENERGY"]
            
            # Calculate average performance by category
            cyclical_performance = 0
            cyclical_count = 0
            
            defensive_performance = 0
            defensive_count = 0
            
            tech_performance = 0
            tech_count = 0
            
            energy_performance = 0
            energy_count = 0
            
            for sector, data in sector_data.items():
                daily_change = data.get('daily_change', 0)
                
                if sector in cyclical_sectors:
                    cyclical_performance += daily_change
                    cyclical_count += 1
                elif sector in defensive_sectors:
                    defensive_performance += daily_change
                    defensive_count += 1
                elif sector in technology_sectors:
                    tech_performance += daily_change
                    tech_count += 1
                elif sector in energy_sectors:
                    energy_performance += daily_change
                    energy_count += 1
                    
            # Calculate averages
            cyclical_avg = cyclical_performance / cyclical_count if cyclical_count > 0 else 0
            defensive_avg = defensive_performance / defensive_count if defensive_count > 0 else 0
            tech_avg = tech_performance / tech_count if tech_count > 0 else 0
            energy_avg = energy_performance / energy_count if energy_count > 0 else 0
            
            # Determine pattern
            if cyclical_avg > 1 and defensive_avg < 0:
                pattern = "risk-on"
            elif defensive_avg > 1 and cyclical_avg < 0:
                pattern = "risk-off"
            elif cyclical_avg > 0 and defensive_avg > 0 and tech_avg > 0:
                pattern = "broad advance"
            elif cyclical_avg < 0 and defensive_avg < 0 and tech_avg < 0:
                pattern = "broad decline"
            elif tech_avg > 1 and cyclical_avg < 0:
                pattern = "tech-led"
            elif energy_avg > 1 and (cyclical_avg + defensive_avg + tech_avg) / 3 < 0:
                pattern = "energy-led"
            else:
                pattern = "mixed"
                
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error identifying rotation pattern: {e}")
            return "unknown"
            
    def _calculate_relative_strength(self, sector_data):
        """
        Calculate relative strength of sectors.
        
        Args:
            sector_data (dict): Sector performance data
            
        Returns:
            dict: Relative strength data
        """
        try:
            if not sector_data:
                return {}
                
            # Calculate market average
            market_avg = sum(data.get('daily_change', 0) for data in sector_data.values()) / len(sector_data) if sector_data else 0
            
            # Calculate relative strength
            relative_strength = {}
            
            for sector, data in sector_data.items():
                daily_change = data.get('daily_change', 0)
                relative = daily_change - market_avg
                
                relative_strength[sector] = {
                    "actual": daily_change,
                    "relative": relative,
                    "rs_factor": daily_change / market_avg if market_avg != 0 else 0
                }
                
            return relative_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating relative strength: {e}")
            return {}
            
    def _identify_price_patterns(self, nifty_data):
        """
        Identify price patterns from NIFTY data.
        
        Args:
            nifty_data (list): NIFTY price data
            
        Returns:
            dict: Price pattern data
        """
        try:
            if not nifty_data or len(nifty_data) < 2:
                return {"pattern": "unknown"}
                
            # Get today's and yesterday's data
            today = nifty_data[-1]
            yesterday = nifty_data[-2]
            
            # Check for gap
            open_gap = (today['open'] - yesterday['close']) / yesterday['close'] * 100
            
            # Check for day pattern
            if today['close'] > today['open']:
                # Bullish day
                body_size = (today['close'] - today['open']) / today['open'] * 100
                upper_shadow = (today['high'] - today['close']) / today['open'] * 100
                lower_shadow = (today['open'] - today['low']) / today['open'] * 100
                
                if body_size > 0.8:
                    day_pattern = "strong bullish"
                elif body_size < 0.2:
                    if upper_shadow > body_size * 2 and lower_shadow > body_size * 2:
                        day_pattern = "doji"
                    else:
                        day_pattern = "weak bullish"
                else:
                    day_pattern = "moderate bullish"
            else:
                # Bearish day
                body_size = (today['open'] - today['close']) / today['open'] * 100
                upper_shadow = (today['high'] - today['open']) / today['open'] * 100
                lower_shadow = (today['close'] - today['low']) / today['open'] * 100
                
                if body_size > 0.8:
                    day_pattern = "strong bearish"
                elif body_size < 0.2:
                    if upper_shadow > body_size * 2 and lower_shadow > body_size * 2:
                        day_pattern = "doji"
                    else:
                        day_pattern = "weak bearish"
                else:
                    day_pattern = "moderate bearish"
                    
            # Check for trend pattern
            if len(nifty_data) >= 5:
                closes = [d['close'] for d in nifty_data]
                
                # Calculate linear regression
                x = np.arange(len(closes))
                y = np.array(closes)
                
                if len(x) == len(y) and len(x) > 0:
                    slope, _ = np.polyfit(x, y, 1)
                    
                    if slope > 0:
                        trend = "uptrend"
                    else:
                        trend = "downtrend"
                        
                    # Calculate standard deviation
                    std_dev = np.std(y)
                    
                    # Calculate regression line
                    reg_line = slope * x + _
                    
                    # Calculate distance from regression line
                    distances = y - reg_line
                    
                    # Check for breakout
                    if distances[-1] > std_dev * 1.5:
                        trend_pattern = "upside breakout"
                    elif distances[-1] < -std_dev * 1.5:
                        trend_pattern = "downside breakout"
                    # Check for reversal
                    elif (trend == "uptrend" and day_pattern.startswith("strong bearish")) or \
                         (trend == "downtrend" and day_pattern.startswith("strong bullish")):
                        trend_pattern = "potential reversal"
                    else:
                        trend_pattern = f"continuing {trend}"
                else:
                    trend_pattern = "unknown"
            else:
                trend_pattern = "unknown"
                
            # Create price pattern data
            pattern_data = {
                "day_pattern": day_pattern,
                "open_gap": open_gap,
                "gap_type": "up gap" if open_gap > 0.2 else ("down gap" if open_gap < -0.2 else "no significant gap"),
                "body_size": body_size,
                "trend_pattern": trend_pattern
            }
            
            return pattern_data
            
        except Exception as e:
            self.logger.error(f"Error identifying price patterns: {e}")
            return {"pattern": "unknown"}
            
    def _identify_volume_patterns(self, nifty_data):
        """
        Identify volume patterns from NIFTY data.
        
        Args:
            nifty_data (list): NIFTY price data
            
        Returns:
            dict: Volume pattern data
        """
        try:
            if not nifty_data or len(nifty_data) < 2:
                return {"pattern": "unknown"}
                
            # Get today's and yesterday's data
            today = nifty_data[-1]
            yesterday = nifty_data[-2]
            
            # Calculate volume change
            volume_change = (today['volume'] - yesterday['volume']) / yesterday['volume'] * 100 if yesterday['volume'] > 0 else 0
            
            # Calculate average volume (last 5 days)
            avg_volume = sum(d['volume'] for d in nifty_data[-5:]) / len(nifty_data[-5:]) if len(nifty_data) >= 5 else today['volume']
            
            # Calculate volume ratio
            volume_ratio = today['volume'] / avg_volume if avg_volume > 0 else 1.0
            
            # Check price-volume relationship
            price_change = (today['close'] - yesterday['close']) / yesterday['close'] * 100 if yesterday['close'] > 0 else 0
            
            if volume_ratio > 1.5:
                if price_change > 0.5:
                    relationship = "high volume advance"
                elif price_change < -0.5:
                    relationship = "high volume decline"
                else:
                    relationship = "high volume consolidation"
            elif volume_ratio < 0.7:
                if price_change > 0.5:
                    relationship = "low volume advance"
                elif price_change < -0.5:
                    relationship = "low volume decline"
                else:
                    relationship = "low volume consolidation"
            else:
                if price_change > 0.5:
                    relationship = "normal volume advance"
                elif price_change < -0.5:
                    relationship = "normal volume decline"
                else:
                    relationship = "normal volume consolidation"
                    
            # Check for volume climax
            if volume_ratio > 2.0 and abs(price_change) > 1.5:
                climax = "potential volume climax"
            else:
                climax = "no climax detected"
                
            # Create volume pattern data
            pattern_data = {
                "volume_change": volume_change,
                "volume_ratio": volume_ratio,
                "relationship": relationship,
                "climax": climax,
                "pattern": relationship
            }
            
            return pattern_data
            
        except Exception as e:
            self.logger.error(f"Error identifying volume patterns: {e}")
            return {"pattern": "unknown"}
            
    def _identify_technical_patterns(self, nifty_data):
        """
        Identify technical patterns from NIFTY data.
        
        Args:
            nifty_data (list): NIFTY price data
            
        Returns:
            dict: Technical pattern data
        """
        try:
            if not nifty_data or len(nifty_data) < 10:
                return {"pattern": "unknown"}
                
            # Convert to pandas DataFrame
            df = pd.DataFrame(nifty_data)
            
            # Calculate technical indicators
            # 1. Moving Averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # 2. RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 3. MACD
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # Check for patterns
            patterns = []
            
            # Check Moving Average Crossover
            if len(df) >= 2:
                if df['sma_5'].iloc[-2] < df['sma_20'].iloc[-2] and df['sma_5'].iloc[-1] > df['sma_20'].iloc[-1]:
                    patterns.append("bullish MA crossover")
                elif df['sma_5'].iloc[-2] > df['sma_20'].iloc[-2] and df['sma_5'].iloc[-1] < df['sma_20'].iloc[-1]:
                    patterns.append("bearish MA crossover")
                    
            # Check RSI levels
            if not df['rsi'].iloc[-1].isna():
                rsi = df['rsi'].iloc[-1]
                
                if rsi > 70:
                    patterns.append("overbought RSI")
                elif rsi < 30:
                    patterns.append("oversold RSI")
                    
            # Check MACD crossover
            if len(df) >= 2 and not df['macd'].iloc[-1].isna() and not df['signal'].iloc[-1].isna():
                if df['macd'].iloc[-2] < df['signal'].iloc[-2] and df['macd'].iloc[-1] > df['signal'].iloc[-1]:
                    patterns.append("bullish MACD crossover")
                elif df['macd'].iloc[-2] > df['signal'].iloc[-2] and df['macd'].iloc[-1] < df['signal'].iloc[-1]:
                    patterns.append("bearish MACD crossover")
                    
            # Check for support/resistance
            if len(df) >= 5:
                current_close = df['close'].iloc[-1]
                
                # Find recent highs and lows
                recent_high = df['high'].iloc[-5:].max()
                recent_low = df['low'].iloc[-5:].min()
                
                # Check for breakout or bounce
                if current_close > recent_high * 0.99 and df['close'].iloc[-2] < recent_high * 0.99:
                    patterns.append("resistance breakout")
                elif current_close < recent_low * 1.01 and df['close'].iloc[-2] > recent_low * 1.01:
                    patterns.append("support breakdown")
                elif abs(current_close - recent_low) / recent_low < 0.01:
                    patterns.append("support bounce")
                elif abs(current_close - recent_high) / recent_high < 0.01:
                    patterns.append("resistance rejection")
                    
            # Determine primary pattern
            if patterns:
                primary_pattern = patterns[0]
            else:
                primary_pattern = "no clear technical pattern"
                
            # Create technical pattern data
            pattern_data = {
                "patterns": patterns,
                "primary_pattern": primary_pattern,
                "rsi": df['rsi'].iloc[-1] if not df['rsi'].iloc[-1].isna() else None,
                "macd": df['macd'].iloc[-1] if not df['macd'].iloc[-1].isna() else None,
                "signal": df['signal'].iloc[-1] if not df['signal'].iloc[-1].isna() else None,
                "sma_5": df['sma_5'].iloc[-1] if not df['sma_5'].iloc[-1].isna() else None,
                "sma_20": df['sma_20'].iloc[-1] if not df['sma_20'].iloc[-1].isna() else None
            }
            
            return pattern_data
            
        except Exception as e:
            self.logger.error(f"Error identifying technical patterns: {e}")
            return {"pattern": "unknown"}
            
    def _determine_overall_pattern(self, price_patterns, volume_patterns, technical_patterns):
        """
        Determine overall market pattern.
        
        Args:
            price_patterns (dict): Price pattern data
            volume_patterns (dict): Volume pattern data
            technical_patterns (dict): Technical pattern data
            
        Returns:
            str: Overall pattern
        """
        try:
            patterns = []
            
            # Add price pattern
            day_pattern = price_patterns.get('day_pattern', '')
            trend_pattern = price_patterns.get('trend_pattern', '')
            
            if day_pattern:
                patterns.append(day_pattern)
                
            if trend_pattern:
                patterns.append(trend_pattern)
                
            # Add volume pattern
            volume_pattern = volume_patterns.get('relationship', '')
            
            if volume_pattern:
                patterns.append(volume_pattern)
                
            # Add technical patterns
            tech_patterns = technical_patterns.get('patterns', [])
            
            if tech_patterns:
                patterns.extend(tech_patterns)
                
            # Count pattern types
            bullish_count = sum(1 for p in patterns if 'bullish' in p or 'advance' in p or 'breakout' in p)
            bearish_count = sum(1 for p in patterns if 'bearish' in p or 'decline' in p or 'breakdown' in p)
            
            # Determine overall pattern
            if bullish_count > bearish_count + 1:
                if 'high volume' in ' '.join(patterns):
                    overall = "strong bullish"
                else:
                    overall = "bullish"
            elif bearish_count > bullish_count + 1:
                if 'high volume' in ' '.join(patterns):
                    overall = "strong bearish"
                else:
                    overall = "bearish"
            elif 'reversal' in ' '.join(patterns):
                if bullish_count > bearish_count:
                    overall = "potential bullish reversal"
                else:
                    overall = "potential bearish reversal"
            elif 'consolidation' in ' '.join(patterns) or 'doji' in ' '.join(patterns):
                overall = "consolidation"
            else:
                overall = "mixed signals"
                
            return overall
            
        except Exception as e:
            self.logger.error(f"Error determining overall pattern: {e}")
            return "unknown"
            
    def _is_prediction_correct(self, prediction):
        """
        Check if a prediction was correct.
        
        Args:
            prediction (dict): Prediction data
            
        Returns:
            bool: True if prediction was correct, False otherwise
        """
        try:
            symbol = prediction.get('symbol')
            exchange = prediction.get('exchange')
            predicted_direction = prediction.get('prediction')
            
            # Get actual data
            actual_data = self._get_daily_data(symbol, exchange)
            
            if not actual_data:
                return False
                
            # Determine actual direction
            if actual_data.get('close', 0) > actual_data.get('prev_close', 0):
                actual_direction = "up"
            else:
                actual_direction = "down"
                
            # Check if prediction was correct
            return predicted_direction == actual_direction
            
        except Exception as e:
            self.logger.error(f"Error checking prediction correctness: {e}")
            return False