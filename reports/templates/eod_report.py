# reports/templates/eod_report.py (Session 47: Report Templates & Formatters)

import logging
from datetime import datetime, timedelta

class EODReport:
    """
    End-of-day report template for daily performance analysis.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the EOD report generator.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("EOD report template initialized")
    
    def generate_report(self):
        """
        Generate end-of-day report.
        
        Returns:
            dict: Report data
        """
        try:
            self.logger.info("Generating EOD report")
            
            # Get today's date
            today = datetime.now().date()
            today_str = today.strftime("%A, %B %d, %Y")
            
            # Check if market was open today
            from trading.market_hours import MarketHours
            market_hours = MarketHours()
            
            was_trading_day = market_hours.was_trading_day(today)
            
            if not was_trading_day:
                # Generate non-trading day report
                report_data = {
                    "date": today,
                    "was_trading_day": False,
                    "message": f"Today ({today_str}) was not a trading day. Markets were closed."
                }
                
                return report_data
            
            # Get market summary
            market_summary = self._get_market_summary()
            
            # Get trading performance
            trading_performance = self._get_trading_performance()
            
            # Get sector performance
            sector_performance = self._get_sector_performance()
            
            # Get prediction performance
            prediction_performance = self._get_prediction_performance()
            
            # Get market analysis
            market_analysis = self._get_market_analysis()
            
            # Create report data
            report_data = {
                "date": today,
                "was_trading_day": True,
                "title": f"End-of-Day Report - {today_str}",
                "market_summary": market_summary,
                "trading_performance": trading_performance,
                "sector_performance": sector_performance,
                "prediction_performance": prediction_performance,
                "market_analysis": market_analysis
            }
            
            # Save report to database
            if self.db:
                report_id = f"eod_report_{today.strftime('%Y%m%d')}"
                
                self.db.reports.update_one(
                    {"report_id": report_id},
                    {"$set": {
                        "report_id": report_id,
                        "type": "eod_report",
                        "date": datetime.now(),
                        "data": report_data
                    }},
                    upsert=True
                )
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating EOD report: {e}")
            return {"error": str(e)}
    
    def _get_market_summary(self):
        """
        Get market summary.
        
        Returns:
            dict: Market summary
        """
        try:
            if not self.db:
                return {}
            
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            # Get market indices data
            indices = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
            indices_data = {}
            
            for index in indices:
                # Get today's data
                data = self.db.market_data.find_one({
                    'symbol': index,
                    'exchange': 'NSE',
                    'timeframe': 'day',
                    'timestamp': {'$gte': today, '$lt': tomorrow}
                })
                
                if data:
                    # Get previous day's data
                    yesterday = today - timedelta(days=1)
                    
                    prev_data = self.db.market_data.find_one({
                        'symbol': index,
                        'exchange': 'NSE',
                        'timeframe': 'day',
                        'timestamp': {'$gte': yesterday, '$lt': today}
                    })
                    
                    prev_close = prev_data.get('close', 0) if prev_data else 0
                    
                    # Calculate change
                    if prev_close > 0:
                        change = ((data.get('close', 0) - prev_close) / prev_close) * 100
                    else:
                        change = 0
                    
                    # Create index data
                    indices_data[index] = {
                        'open': data.get('open', 0),
                        'high': data.get('high', 0),
                        'low': data.get('low', 0),
                        'close': data.get('close', 0),
                        'prev_close': prev_close,
                        'change': change
                    }
            
            # Get market breadth data
            breadth = self.db.market_breadth.find_one({
                'date': {'$gte': today, '$lt': tomorrow}
            })
            
            breadth_data = {}
            
            if breadth:
                advances = breadth.get('advances', 0)
                declines = breadth.get('declines', 0)
                unchanged = breadth.get('unchanged', 0)
                
                total = advances + declines + unchanged
                
                if total > 0:
                    advance_percent = (advances / total) * 100
                    decline_percent = (declines / total) * 100
                else:
                    advance_percent = 0
                    decline_percent = 0
                
                breadth_data = {
                    'advances': advances,
                    'declines': declines,
                    'unchanged': unchanged,
                    'advance_percent': advance_percent,
                    'decline_percent': decline_percent
                }
            
            # Determine market direction
            nifty_change = indices_data.get('NIFTY', {}).get('change', 0)
            
            if nifty_change > 1.0:
                direction = "strongly positive"
            elif nifty_change > 0.2:
                direction = "positive"
            elif nifty_change < -1.0:
                direction = "strongly negative"
            elif nifty_change < -0.2:
                direction = "negative"
            else:
                direction = "flat"
            
            # Get volume comparison with previous day
            nifty_volume = indices_data.get('NIFTY', {}).get('volume', 0)
            nifty_prev_volume = 0
            
            prev_nifty = self.db.market_data.find_one({
                'symbol': 'NIFTY',
                'exchange': 'NSE',
                'timeframe': 'day',
                'timestamp': {'$gte': today - timedelta(days=1), '$lt': today}
            })
            
            if prev_nifty:
                nifty_prev_volume = prev_nifty.get('volume', 0)
            
            volume_change = ((nifty_volume - nifty_prev_volume) / nifty_prev_volume) * 100 if nifty_prev_volume > 0 else 0
            
            # Create market summary
            summary = {
                'direction': direction,
                'nifty': indices_data.get('NIFTY', {}),
                'banknifty': indices_data.get('BANKNIFTY', {}),
                'finnifty': indices_data.get('FINNIFTY', {}),
                'breadth': breadth_data,
                'volume_change': volume_change
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting market summary: {e}")
            return {}
    
    def _get_trading_performance(self):
        """
        Get trading performance.
        
        Returns:
            dict: Trading performance
        """
        try:
            if not self.db:
                return {}
            
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            # Get today's trades
            trades = list(self.db.trades.find({
                "$or": [
                    {"entry_time": {"$gte": today, "$lt": tomorrow}},
                    {"exit_time": {"$gte": today, "$lt": tomorrow}}
                ]
            }))
            
            # Initialize metrics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]
            
            num_winning = len(winning_trades)
            num_losing = len(losing_trades)
            
            win_rate = (num_winning / total_trades) * 100 if total_trades > 0 else 0
            
            total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
            total_loss = sum(t.get('profit_loss', 0) for t in losing_trades)
            net_pnl = total_profit + total_loss
            
            # Calculate average metrics
            avg_win = total_profit / num_winning if num_winning > 0 else 0
            avg_loss = total_loss / num_losing if num_losing > 0 else 0
            
            # Calculate profit factor
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf') if total_profit > 0 else 0
            
            # Analyze by strategy
            strategy_performance = {}
            
            for trade in trades:
                strategy = trade.get('strategy', 'unknown')
                
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        'trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'profit_loss': 0
                    }
                
                perf = strategy_performance[strategy]
                perf['trades'] += 1
                
                if trade.get('profit_loss', 0) > 0:
                    perf['wins'] += 1
                else:
                    perf['losses'] += 1
                
                perf['profit_loss'] += trade.get('profit_loss', 0)
            
            # Calculate win rate for each strategy
            for strategy, perf in strategy_performance.items():
                if perf['trades'] > 0:
                    perf['win_rate'] = (perf['wins'] / perf['trades']) * 100
                else:
                    perf['win_rate'] = 0
            
            # Get top and bottom strategies
            if strategy_performance:
                sorted_strategies = sorted(
                    strategy_performance.items(),
                    key=lambda x: x[1]['profit_loss'],
                    reverse=True
                )
                
                top_strategy = sorted_strategies[0][0] if sorted_strategies else None
                bottom_strategy = sorted_strategies[-1][0] if len(sorted_strategies) > 1 else None
            else:
                top_strategy = None
                bottom_strategy = None
            
            # Create trading performance data
            performance = {
                'total_trades': total_trades,
                'winning_trades': num_winning,
                'losing_trades': num_losing,
                'win_rate': win_rate,
                'net_pnl': net_pnl,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'strategy_performance': strategy_performance,
                'top_strategy': top_strategy,
                'bottom_strategy': bottom_strategy
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error getting trading performance: {e}")
            return {}
    
    def _get_sector_performance(self):
        """
        Get sector performance.
        
        Returns:
            dict: Sector performance
        """
        try:
            if not self.db:
                return {}
            
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            # Get sector data
            sectors = list(self.db.sector_performance.find({
                'date': {'$gte': today, '$lt': tomorrow}
            }))
            
            if not sectors:
                return {
                    'count': 0,
                    'message': "No sector data available"
                }
            
            # Sort by performance
            sorted_sectors = sorted(sectors, key=lambda x: x.get('change', 0), reverse=True)
            
            top_sectors = sorted_sectors[:3]
            bottom_sectors = sorted_sectors[-3:]
            
            # Get sector rotation analysis
            from ml.prediction.sector_rotation_analyzer import SectorRotationAnalyzer
            analyzer = SectorRotationAnalyzer(self.db)
            
            rotation = analyzer.analyze_daily_rotation()
            
            # Create sector performance data
            performance = {
                'count': len(sectors),
                'sectors': sorted_sectors,
                'top_sectors': [{'name': s.get('name', 'Unknown'), 'change': s.get('change', 0)} for s in top_sectors],
                'bottom_sectors': [{'name': s.get('name', 'Unknown'), 'change': s.get('change', 0)} for s in bottom_sectors],
                'rotation_pattern': rotation.get('rotation_pattern', 'unknown'),
                'outperforming_sectors': rotation.get('outperforming_sectors', []),
                'underperforming_sectors': rotation.get('underperforming_sectors', [])
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error getting sector performance: {e}")
            return {}
    
    def _get_prediction_performance(self):
        """
        Get prediction performance.
        
        Returns:
            dict: Prediction performance
        """
        try:
            if not self.db:
                return {}
            
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday = today - timedelta(days=1)
            
            # Get validation results for daily predictions
            validation_id = f"daily_{today.strftime('%Y%m%d')}"
            
            validation = self.db.prediction_validations.find_one({
                'validation_id': validation_id
            })
            
            if not validation:
                return {
                    'status': 'no_data',
                    'message': "No prediction validation data available"
                }
            
            # Extract validation metrics
            total_predictions = validation.get('predictions', 0)
            correct_predictions = validation.get('correct', 0)
            accuracy = validation.get('accuracy', 0)
            
            # Get confidence level data
            confidence_results = validation.get('confidence_results', {})
            
            high_conf = confidence_results.get('high', {})
            med_conf = confidence_results.get('medium', {})
            low_conf = confidence_results.get('low', {})
            
            # Get direction performance
            details = validation.get('details', [])
            
            up_predictions = [d for d in details if d.get('predicted') == 'up']
            down_predictions = [d for d in details if d.get('predicted') == 'down']
            
            up_correct = [d for d in up_predictions if d.get('correct', False)]
            down_correct = [d for d in down_predictions if d.get('correct', False)]
            
            up_accuracy = (len(up_correct) / len(up_predictions)) * 100 if up_predictions else 0
            down_accuracy = (len(down_correct) / len(down_predictions)) * 100 if down_predictions else 0
            
            # Compare to historical accuracy
            from automation.prediction_validator import PredictionValidator
            validator = PredictionValidator(self.db)
            
            historical = validator.analyze_prediction_performance(days=30)
            
            historical_accuracy = historical.get('overall', {}).get('accuracy', 0) if isinstance(historical, dict) else 0
            
            # Create prediction performance data
            performance = {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'confidence_levels': {
                    'high': high_conf,
                    'medium': med_conf,
                    'low': low_conf
                },
                'direction_performance': {
                    'up': {
                        'total': len(up_predictions),
                        'correct': len(up_correct),
                        'accuracy': up_accuracy
                    },
                    'down': {
                        'total': len(down_predictions),
                        'correct': len(down_correct),
                        'accuracy': down_accuracy
                    }
                },
                'historical_comparison': {
                    'historical_accuracy': historical_accuracy,
                    'difference': accuracy - historical_accuracy
                }
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error getting prediction performance: {e}")
            return {}
    
    def _get_market_analysis(self):
        """
        Get market analysis.
        
        Returns:
            dict: Market analysis
        """
        try:
            if not self.db:
                return {}
            
            # Get end-of-day analysis
            from automation.eod_analyzer import EODAnalyzer
            analyzer = EODAnalyzer(self.db)
            
            # Run analysis if not already done
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            analysis_id = f"eod_analysis_{today.strftime('%Y%m%d')}"
            
            analysis = self.db.eod_analysis.find_one({
                'analysis_id': analysis_id
            })
            
            if not analysis:
                # Run analysis and get results
                analysis_results = analyzer.analyze()
                
                # Store in database
                self.db.eod_analysis.insert_one({
                    'analysis_id': analysis_id,
                    'date': today,
                    'results': analysis_results
                })
                
                return analysis_results
            else:
                return analysis.get('results', {})
            
        except Exception as e:
            self.logger.error(f"Error getting market analysis: {e}")
            return {}