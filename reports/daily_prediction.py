# reports/daily_prediction.py (Session 46: Report Generation System)

import logging
from datetime import datetime, timedelta

class DailyPredictionReport:
    """
    Generates daily prediction reports.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the daily prediction report generator.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Daily prediction report generator initialized")
    
    def generate_report(self):
        """
        Generate daily prediction report.
        
        Returns:
            dict: Report data
        """
        try:
            self.logger.info("Generating daily prediction report")
            
            # Get market data
            market_data = self._get_market_data()
            
            # Get prediction data
            prediction_data = self._get_prediction_data()
            
            # Get sector data
            sector_data = self._get_sector_data()
            
            # Generate market summary
            market_summary = self._generate_market_summary(market_data)
            
            # Generate prediction summary
            prediction_summary = self._generate_prediction_summary(prediction_data)
            
            # Generate sector summary
            sector_summary = self._generate_sector_summary(sector_data)
            
            # Generate top opportunities
            top_opportunities = self._generate_top_opportunities(prediction_data)
            
            # Create report data
            report_data = {
                'date': datetime.now(),
                'market_summary': market_summary,
                'prediction_summary': prediction_summary,
                'sector_summary': sector_summary,
                'top_opportunities': top_opportunities,
                'market_data': market_data,
                'prediction_data': prediction_data,
                'sector_data': sector_data
            }
            
            # Store report in database
            if self.db:
                report_id = f"daily_prediction_{datetime.now().strftime('%Y%m%d')}"
                
                self.db.reports.update_one(
                    {'report_id': report_id},
                    {'$set': {
                        'report_id': report_id,
                        'type': 'daily_prediction',
                        'date': datetime.now(),
                        'data': report_data
                    }},
                    upsert=True
                )
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating daily prediction report: {e}")
            return {"error": str(e)}
    
    def _get_market_data(self):
        """
        Get market data for the report.
        
        Returns:
            dict: Market data
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
            
            # Create market data
            market_data = {
                'indices': indices_data,
                'breadth': breadth_data
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}
    
    def _get_prediction_data(self):
        """
        Get prediction data for the report.
        
        Returns:
            dict: Prediction data
        """
        try:
            if not self.db:
                return {}
            
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            # Get today's predictions
            cursor = self.db.predictions.find({
                'date': {'$gte': today, '$lt': tomorrow},
                'prediction_type': 'daily'
            })
            
            predictions = list(cursor)
            
            # Get yesterday's prediction validation results
            yesterday = today - timedelta(days=1)
            
            validation_id = f"daily_{yesterday.strftime('%Y%m%d')}"
            
            validation = self.db.prediction_validations.find_one({
                'validation_id': validation_id
            })
            
            # Create prediction data
            prediction_data = {
                'predictions': predictions,
                'validation': validation
            }
            
            return prediction_data
            
        except Exception as e:
            self.logger.error(f"Error getting prediction data: {e}")
            return {}
    
    def _get_sector_data(self):
        """
        Get sector data for the report.
        
        Returns:
            dict: Sector data
        """
        try:
            if not self.db:
                return {}
            
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            # Get sector performance data
            cursor = self.db.sector_performance.find({
                'date': {'$gte': today, '$lt': tomorrow}
            })
            
            sectors = list(cursor)
            
            # Get sector rotation analysis
            from ml.prediction.sector_rotation_analyzer import SectorRotationAnalyzer
            analyzer = SectorRotationAnalyzer(self.db)
            
            rotation = analyzer.analyze_daily_rotation()
            
            # Create sector data
            sector_data = {
                'sectors': sectors,
                'rotation': rotation
            }
            
            return sector_data
            
        except Exception as e:
            self.logger.error(f"Error getting sector data: {e}")
            return {}
    
    def _generate_market_summary(self, market_data):
        """
        Generate market summary.
        
        Args:
            market_data (dict): Market data
            
        Returns:
            dict: Market summary
        """
        try:
            indices = market_data.get('indices', {})
            breadth = market_data.get('breadth', {})
            
            # Determine market direction
            nifty_change = indices.get('NIFTY', {}).get('change', 0)
            
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
            
            # Analyze breadth
            advance_percent = breadth.get('advance_percent', 0)
            
            if advance_percent > 60:
                breadth_type = "strongly positive"
            elif advance_percent > 55:
                breadth_type = "positive"
            elif advance_percent < 40:
                breadth_type = "strongly negative"
            elif advance_percent < 45:
                breadth_type = "negative"
            else:
                breadth_type = "neutral"
            
            # Create summary
            summary = {
                'direction': direction,
                'breadth': breadth_type,
                'nifty': indices.get('NIFTY', {}),
                'banknifty': indices.get('BANKNIFTY', {}),
                'finnifty': indices.get('FINNIFTY', {}),
                'advances': breadth.get('advances', 0),
                'declines': breadth.get('declines', 0)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating market summary: {e}")
            return {}
    
    def _generate_prediction_summary(self, prediction_data):
        """
        Generate prediction summary.
        
        Args:
            prediction_data (dict): Prediction data
            
        Returns:
            dict: Prediction summary
        """
        try:
            predictions = prediction_data.get('predictions', [])
            validation = prediction_data.get('validation', {})
            
            if not predictions:
                return {
                    'count': 0,
                    'message': "No predictions available"
                }
            
            # Count predictions by direction
            up_predictions = [p for p in predictions if p.get('prediction') == 'up']
            down_predictions = [p for p in predictions if p.get('prediction') == 'down']
            
            # Count predictions by confidence
            high_conf = [p for p in predictions if p.get('confidence', 0) >= 0.7]
            med_conf = [p for p in predictions if 0.5 <= p.get('confidence', 0) < 0.7]
            low_conf = [p for p in predictions if p.get('confidence', 0) < 0.5]
            
            # Previous day accuracy
            prev_accuracy = validation.get('accuracy', 0) if validation else 0
            
            # Create summary
            # Create summary
            summary = {
                'count': len(predictions),
                'up_count': len(up_predictions),
                'down_count': len(down_predictions),
                'high_confidence': len(high_conf),
                'med_confidence': len(med_conf),
                'low_confidence': len(low_conf),
                'yesterday_accuracy': prev_accuracy,
                'bullish_bias': len(up_predictions) > len(down_predictions),
                'bearish_bias': len(down_predictions) > len(up_predictions),
                'sentiment': 'bullish' if len(up_predictions) > len(down_predictions) else ('bearish' if len(down_predictions) > len(up_predictions) else 'neutral')
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating prediction summary: {e}")
            return {}
    
    def _generate_sector_summary(self, sector_data):
        """
        Generate sector summary.
        
        Args:
            sector_data (dict): Sector data
            
        Returns:
            dict: Sector summary
        """
        try:
            sectors = sector_data.get('sectors', [])
            rotation = sector_data.get('rotation', {})
            
            if not sectors and not rotation:
                return {
                    'count': 0,
                    'message': "No sector data available"
                }
            
            # Get top and bottom performing sectors
            if sectors:
                # Sort by performance
                sorted_sectors = sorted(sectors, key=lambda x: x.get('change', 0), reverse=True)
                
                top_sectors = sorted_sectors[:3]
                bottom_sectors = sorted_sectors[-3:]
            else:
                top_sectors = []
                bottom_sectors = []
            
            # Get rotation pattern
            rotation_pattern = rotation.get('rotation_pattern', 'unknown')
            
            # Create summary
            summary = {
                'count': len(sectors),
                'top_sectors': [{'name': s.get('name', 'Unknown'), 'change': s.get('change', 0)} for s in top_sectors],
                'bottom_sectors': [{'name': s.get('name', 'Unknown'), 'change': s.get('change', 0)} for s in bottom_sectors],
                'rotation_pattern': rotation_pattern,
                'outperforming_sectors': rotation.get('outperforming_sectors', []),
                'underperforming_sectors': rotation.get('underperforming_sectors', [])
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating sector summary: {e}")
            return {}
    
    def _generate_top_opportunities(self, prediction_data):
        """
        Generate top trading opportunities.
        
        Args:
            prediction_data (dict): Prediction data
            
        Returns:
            dict: Top opportunities
        """
        try:
            predictions = prediction_data.get('predictions', [])
            
            if not predictions:
                return {
                    'count': 0,
                    'message': "No predictions available"
                }
            
            # Filter for high confidence predictions
            high_conf = [p for p in predictions if p.get('confidence', 0) >= 0.7]
            
            if not high_conf:
                # Use medium confidence if no high confidence predictions
                high_conf = [p for p in predictions if p.get('confidence', 0) >= 0.6]
            
            # Sort by confidence
            high_conf.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Get top 5 opportunities
            top_opportunities = high_conf[:5]
            
            # Separate by direction
            top_bullish = [p for p in high_conf if p.get('prediction') == 'up'][:3]  # Top 3 bullish
            top_bearish = [p for p in high_conf if p.get('prediction') == 'down'][:3]  # Top 3 bearish
            
            # Create opportunities data
            opportunities = {
                'count': len(top_opportunities),
                'top': [
                    {
                        'symbol': p.get('symbol', 'Unknown'),
                        'exchange': p.get('exchange', 'Unknown'),
                        'direction': p.get('prediction', 'unknown'),
                        'confidence': p.get('confidence', 0),
                        'target_price': p.get('target_price', 0),
                        'stop_loss': p.get('stop_loss', 0),
                        'supporting_factors': p.get('supporting_factors', [])
                    }
                    for p in top_opportunities
                ],
                'bullish': [
                    {
                        'symbol': p.get('symbol', 'Unknown'),
                        'exchange': p.get('exchange', 'Unknown'),
                        'confidence': p.get('confidence', 0),
                        'target_price': p.get('target_price', 0),
                        'stop_loss': p.get('stop_loss', 0),
                        'supporting_factors': p.get('supporting_factors', [])
                    }
                    for p in top_bullish
                ],
                'bearish': [
                    {
                        'symbol': p.get('symbol', 'Unknown'),
                        'exchange': p.get('exchange', 'Unknown'),
                        'confidence': p.get('confidence', 0),
                        'target_price': p.get('target_price', 0),
                        'stop_loss': p.get('stop_loss', 0),
                        'supporting_factors': p.get('supporting_factors', [])
                    }
                    for p in top_bearish
                ]
            }
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error generating top opportunities: {e}")
            return {}