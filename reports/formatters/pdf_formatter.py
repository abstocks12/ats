# reports/formatters/pdf_formatter.py (Session 47: Report Templates & Formatters)

import logging
from datetime import datetime
import os
import tempfile

class PDFFormatter:
    """
    Formats reports for PDF output.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the PDF formatter.
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("PDF formatter initialized")
    
    def format(self, report_data):
        """
        Format report data for PDF.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            dict: PDF content data
        """
        try:
            self.logger.info("Formatting report for PDF")
            
            # Determine report type
            if 'market_summary' in report_data and 'prediction_summary' in report_data:
                return self.format_daily_prediction(report_data)
            elif 'market_summary' in report_data and 'trading_performance' in report_data:
                return self.format_eod_report(report_data)
            elif 'market_outlook' in report_data and 'global_markets' in report_data:
                return self.format_morning_report(report_data)
            else:
                return self.format_generic_report(report_data)
                
        except Exception as e:
            self.logger.error(f"Error formatting report for PDF: {e}")
            return {"error": str(e)}
    
    def save(self, pdf_data, filename=None):
        """
        Save PDF data to file.
        
        Args:
            pdf_data (dict): PDF content data
            filename (str): Output filename (optional)
            
        Returns:
            str: Path to saved PDF file
        """
        try:
            self.logger.info("Saving PDF report")
            
            # Import required modules
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib import colors
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            except ImportError:
                self.logger.error("ReportLab not installed. Install with 'pip install reportlab'")
                return None
            
            # Create temporary directory if output directory doesn't exist
            output_dir = "reports/output"
            
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except:
                    # Use temp directory if cannot create output directory
                    output_dir = tempfile.gettempdir()
            
            # Set filename if not provided
            if not filename:
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            # Create full path
            filepath = os.path.join(output_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Add custom styles
            styles.add(ParagraphStyle(
                name='Heading1',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12
            ))
            
            styles.add(ParagraphStyle(
                name='Heading2',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=8
            ))
            
            styles.add(ParagraphStyle(
                name='Normal',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6
            ))
            
            # Initialize elements
            elements = []
            
            # Add title
            title = pdf_data.get('title', 'Report')
            elements.append(Paragraph(title, styles['Heading1']))
            elements.append(Spacer(1, 12))
            
            # Add date
            date_str = pdf_data.get('date', datetime.now().strftime("%A, %B %d, %Y"))
            elements.append(Paragraph(f"Generated on: {date_str}", styles['Normal']))
            elements.append(Spacer(1, 12))
            
            # Add sections
            for section in pdf_data.get('sections', []):
                # Add section title
                elements.append(Paragraph(section.get('title', 'Section'), styles['Heading2']))
                elements.append(Spacer(1, 6))
                
                # Add section content
                content_type = section.get('type', 'text')
                content = section.get('content', '')
                
                if content_type == 'text':
                    # Add paragraphs
                    if isinstance(content, list):
                        for paragraph in content:
                            elements.append(Paragraph(paragraph, styles['Normal']))
                            elements.append(Spacer(1, 6))
                    else:
                        elements.append(Paragraph(content, styles['Normal']))
                        elements.append(Spacer(1, 6))
                        
                elif content_type == 'table':
                    # Add table
                    if content:
                        table = Table(content, hAlign='LEFT')
                        
                        # Add table style
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        
                        elements.append(table)
                        elements.append(Spacer(1, 12))
                
                elements.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(elements)
            
            self.logger.info(f"PDF saved to: {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving PDF: {e}")
            return None
    
    def format_daily_prediction(self, report_data):
        """
        Format daily prediction report for PDF.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            dict: PDF content data
        """
        try:
            date_str = report_data.get('date', datetime.now()).strftime("%A, %B %d, %Y")
            
            # Start with basic structure
            pdf_data = {
                'title': f"Daily Prediction Report - {date_str}",
                'date': date_str,
                'sections': []
            }
            
            # Market summary section
            market_summary = report_data.get('market_summary', {})
            
            direction = market_summary.get('direction', 'unknown').title()
            
            market_section = {
                'title': 'Market Summary',
                'type': 'text',
                'content': [
                    f"Today's market is {direction} with {market_summary.get('advances', 0)} advances and {market_summary.get('declines', 0)} declines."
                ]
            }
            
            # Add index data
            indices_content = []
            
            indices = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
            for index in indices:
                index_data = market_summary.get(index.lower(), {})
                change = index_data.get('change', 0)
                
                if change >= 0:
                    indices_content.append(f"{index}: {index_data.get('close', 0):.2f} (+{change:.2f}%)")
                else:
                    indices_content.append(f"{index}: {index_data.get('close', 0):.2f} ({change:.2f}%)")
            
            market_section['content'].append(", ".join(indices_content))
            
            pdf_data['sections'].append(market_section)
            
            # Prediction summary section
            prediction_summary = report_data.get('prediction_summary', {})
            
            count = prediction_summary.get('count', 0)
            up_count = prediction_summary.get('up_count', 0)
            down_count = prediction_summary.get('down_count', 0)
            
            prediction_section = {
                'title': 'Prediction Summary',
                'type': 'text',
                'content': [
                    f"Today we have {count} predictions ({up_count} bullish, {down_count} bearish).",
                    f"Yesterday's prediction accuracy: {prediction_summary.get('yesterday_accuracy', 0):.2f}%"
                ]
            }
            
            pdf_data['sections'].append(prediction_section)
            
            # Sector summary section
            sector_summary = report_data.get('sector_summary', {})
            
            rotation_pattern = sector_summary.get('rotation_pattern', 'unknown')
            
            sector_section = {
                'title': 'Sector Summary',
                'type': 'text',
                'content': [
                    f"Sector rotation pattern: {rotation_pattern}"
                ]
            }
            
            # Add top performing sectors
            top_sectors = sector_summary.get('top_sectors', [])
            
            if top_sectors:
                sector_section['content'].append("Top Performing Sectors:")
                
                sector_content = []
                for sector in top_sectors:
                    name = sector.get('name', 'Unknown')
                    change = sector.get('change', 0)
                    
                    sector_content.append(f"{name}: +{change:.2f}%")
                
                sector_section['content'].append(", ".join(sector_content))
            
            # Add bottom performing sectors
            bottom_sectors = sector_summary.get('bottom_sectors', [])
            
            if bottom_sectors:
                sector_section['content'].append("Underperforming Sectors:")
                
                sector_content = []
                for sector in bottom_sectors:
                    name = sector.get('name', 'Unknown')
                    change = sector.get('change', 0)
                    
                    sector_content.append(f"{name}: {change:.2f}%")
                
                sector_section['content'].append(", ".join(sector_content))
            
            pdf_data['sections'].append(sector_section)
            
            # Top opportunities section
            top_opportunities = report_data.get('top_opportunities', {})
            
            # Bullish opportunities
            bullish = top_opportunities.get('bullish', [])
            
            if bullish:
                bullish_section = {
                    'title': 'Bullish Opportunities',
                    'type': 'table',
                    'content': [['Symbol', 'Confidence', 'Target', 'Stop']]
                }
                
                for opp in bullish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = f"{opp.get('confidence', 0) * 100:.1f}%"
                    target = f"₹{opp.get('target_price', 0):.2f}" if opp.get('target_price', 0) > 0 else 'N/A'
                    stop = f"₹{opp.get('stop_loss', 0):.2f}" if opp.get('stop_loss', 0) > 0 else 'N/A'
                    
                    bullish_section['content'].append([symbol, confidence, target, stop])
                
                pdf_data['sections'].append(bullish_section)
            
            # Bearish opportunities
            bearish = top_opportunities.get('bearish', [])
            
            if bearish:
                bearish_section = {
                    'title': 'Bearish Opportunities',
                    'type': 'table',
                    'content': [['Symbol', 'Confidence', 'Target', 'Stop']]
                }
                
                for opp in bearish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = f"{opp.get('confidence', 0) * 100:.1f}%"
                    target = f"₹{opp.get('target_price', 0):.2f}" if opp.get('target_price', 0) > 0 else 'N/A'
                    stop = f"₹{opp.get('stop_loss', 0):.2f}" if opp.get('stop_loss', 0) > 0 else 'N/A'
                    
                    bearish_section['content'].append([symbol, confidence, target, stop])
                
                pdf_data['sections'].append(bearish_section)
            
            # Footer section
            footer_section = {
                'title': 'Disclaimer',
                'type': 'text',
                'content': [
                    "This report was automatically generated by the Automated Trading System.",
                    "All predictions are based on mathematical models and should be used for informational purposes only.",
                    "Past performance is not indicative of future results."
                ]
            }
            
            pdf_data['sections'].append(footer_section)
            
            return pdf_data
            
        except Exception as e:
            self.logger.error(f"Error formatting daily prediction for PDF: {e}")
            return {"error": str(e)}
    
    def format_eod_report(self, report_data):
        """
        Format EOD report for PDF.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            dict: PDF content data
        """
        try:
            date_str = report_data.get('date', datetime.now()).strftime("%A, %B %d, %Y")
            
            # Start with basic structure
            pdf_data = {
                'title': f"End-of-Day Report - {date_str}",
                'date': date_str,
                'sections': []
            }
            
            # Market summary section
            market_summary = report_data.get('market_summary', {})
            
            direction = market_summary.get('direction', 'unknown').title()
            
            market_section = {
                'title': 'Market Summary',
                'type': 'text',
                'content': [
                    f"Today's market was {direction}."
                ]
            }
            
            # Add index data
            indices_content = []
            
            indices = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
            for index in indices:
                index_data = market_summary.get(index.lower(), {})
                change = index_data.get('change', 0)
                
                if change >= 0:
                    indices_content.append(f"{index}: {index_data.get('close', 0):.2f} (+{change:.2f}%)")
                else:
                    indices_content.append(f"{index}: {index_data.get('close', 0):.2f} ({change:.2f}%)")
            
            market_section['content'].append(", ".join(indices_content))
            
            # Add volume change
            volume_change = market_summary.get('volume_change', 0)
            
            if volume_change > 0:
                market_section['content'].append(f"Volume increased by {volume_change:.2f}% compared to previous day.")
            else:
                market_section['content'].append(f"Volume decreased by {abs(volume_change):.2f}% compared to previous day.")
            
            # Add breadth
            breadth = market_summary.get('breadth', {})
            advances = breadth.get('advances', 0)
            declines = breadth.get('declines', 0)
            
            market_section['content'].append(f"Market breadth: {advances} advances, {declines} declines.")
            
            pdf_data['sections'].append(market_section)
            
            # Trading performance section
            trading_performance = report_data.get('trading_performance', {})
            
            total_trades = trading_performance.get('total_trades', 0)
            
            if total_trades > 0:
                trading_section = {
                    'title': 'Trading Performance',
                    'type': 'table',
                    'content': [
                        ['Metric', 'Value'],
                        ['Total Trades', str(total_trades)],
                        ['Winning Trades', str(trading_performance.get('winning_trades', 0))],
                        ['Losing Trades', str(trading_performance.get('losing_trades', 0))],
                        ['Win Rate', f"{trading_performance.get('win_rate', 0):.2f}%"],
                        ['Net P&L', f"₹{trading_performance.get('net_pnl', 0):.2f}"],
                        ['Profit Factor', f"{trading_performance.get('profit_factor', 0):.2f}"]
                    ]
                }
                
                pdf_data['sections'].append(trading_section)
                
                # Strategy performance
                strategy_performance = trading_performance.get('strategy_performance', {})
                
                if strategy_performance:
                    strategy_section = {
                        'title': 'Strategy Performance',
                        'type': 'table',
                        'content': [['Strategy', 'Trades', 'Win Rate', 'P&L']]
                    }
                    
                    for strategy, perf in strategy_performance.items():
                        row = [
                            strategy,
                            str(perf.get('trades', 0)),
                            f"{perf.get('win_rate', 0):.2f}%",
                            f"₹{perf.get('profit_loss', 0):.2f}"
                        ]
                        
                        strategy_section['content'].append(row)
                    
                    pdf_data['sections'].append(strategy_section)
            else:
                trading_section = {
                    'title': 'Trading Performance',
                    'type': 'text',
                    'content': ["No trades executed today."]
                }
                
                pdf_data['sections'].append(trading_section)
            
            # Prediction performance section
            prediction_performance = report_data.get('prediction_performance', {})
            
            if prediction_performance.get('status') != 'no_data':
                prediction_section = {
                    'title': 'Prediction Performance',
                    'type': 'table',
                    'content': [
                        ['Metric', 'Value'],
                        ['Total Predictions', str(prediction_performance.get('total_predictions', 0))],
                        ['Correct Predictions', str(prediction_performance.get('correct_predictions', 0))],
                        ['Accuracy', f"{prediction_performance.get('accuracy', 0):.2f}%"]
                    ]
                }
                
                # Add direction performance
                direction_performance = prediction_performance.get('direction_performance', {})
                
                if direction_performance:
                    up_perf = direction_performance.get('up', {})
                    down_perf = direction_performance.get('down', {})
                    
                    prediction_section['content'].append(['Up Predictions Accuracy', f"{up_perf.get('accuracy', 0):.2f}%"])
                    prediction_section['content'].append(['Down Predictions Accuracy', f"{down_perf.get('accuracy', 0):.2f}%"])
                
                # Add historical comparison
                historical = prediction_performance.get('historical_comparison', {})
                
                if historical:
                    prediction_section['content'].append(['Historical Accuracy', f"{historical.get('historical_accuracy', 0):.2f}%"])
                    prediction_section['content'].append(['Difference', f"{historical.get('difference', 0):+.2f}%"])
                
                pdf_data['sections'].append(prediction_section)
            
            # Sector performance section
            sector_performance = report_data.get('sector_performance', {})
            
            rotation_pattern = sector_performance.get('rotation_pattern', 'unknown')
            
            sector_section = {
                'title': 'Sector Performance',
                'type': 'text',
                'content': [
                    f"Sector rotation pattern: {rotation_pattern}"
                ]
            }
            
            # Add top and bottom sectors as table
            top_sectors = sector_performance.get('top_sectors', [])
            bottom_sectors = sector_performance.get('bottom_sectors', [])
            
            if top_sectors or bottom_sectors:
                sectors_table = {
                    'title': 'Sector Performance',
                    'type': 'table',
                    'content': [['Sector', 'Change']]
                }
                
                for sector in top_sectors:
                    name = sector.get('name', 'Unknown')
                    change = sector.get('change', 0)
                    
                    sectors_table['content'].append([name, f"+{change:.2f}%"])
                
                for sector in bottom_sectors:
                    name = sector.get('name', 'Unknown')
                    change = sector.get('change', 0)
                    
                    sectors_table['content'].append([name, f"{change:.2f}%"])
                
                pdf_data['sections'].append(sectors_table)
            else:
                pdf_data['sections'].append(sector_section)
            
            # Market analysis section
            market_analysis = report_data.get('market_analysis', {})
            
            # Key observations
            observations = market_analysis.get('observations', [])
            
            if observations:
                observation_section = {
                    'title': 'Key Observations',
                    'type': 'text',
                    'content': []
                }
                
                for observation in observations:
                    observation_section['content'].append(f"• {observation}")
                
                pdf_data['sections'].append(observation_section)
            
            # Recommendations
            recommendations = market_analysis.get('recommendations', [])
            
            if recommendations:
                recommendation_section = {
                    'title': 'Recommendations',
                    'type': 'text',
                    'content': []
                }
                
                for recommendation in recommendations:
                    recommendation_section['content'].append(f"• {recommendation}")
                
                pdf_data['sections'].append(recommendation_section)
            
            # Footer section
            footer_section = {
                'title': 'Disclaimer',
                'type': 'text',
                'content': [
                    "This report was automatically generated by the Automated Trading System.",
                    "All analysis is based on historical data and should be used for informational purposes only.",
                    "Past performance is not indicative of future results."
                ]
            }
            
            pdf_data['sections'].append(footer_section)
            
            return pdf_data
            
        except Exception as e:
            self.logger.error(f"Error formatting EOD report for PDF: {e}")
            return {"error": str(e)}
    
    def format_morning_report(self, report_data):
        """
        Format morning report for PDF.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            dict: PDF content data
        """
        try:
            date_str = report_data.get('date', datetime.now()).strftime("%A, %B %d, %Y")
            
            # Check if trading day
            if not report_data.get('is_trading_day', True):
                return {
                    'title': f"Morning Report - {date_str}",
                    'date': date_str,
                    'sections': [
                        {
                            'title': 'Market Status',
                            'type': 'text',
                            'content': [report_data.get('message', 'Markets are closed today.')]
                        }
                    ]
                }
            
            # Start with basic structure
            pdf_data = {
                'title': f"Morning Report - {date_str}",
                'date': date_str,
                'sections': []
            }
            
            # Market outlook section
            market_outlook = report_data.get('market_outlook', {})
            
            outlook = market_outlook.get('outlook', 'neutral').title()
            description = market_outlook.get('description', '')
            
            outlook_section = {
                'title': 'Market Outlook',
                'type': 'text',
                'content': [
                    f"Outlook: {outlook}",
                    description
                ]
            }
            
            # Add factors
            factors = market_outlook.get('factors', [])
            
            if factors:
                outlook_section['content'].append("Key factors:")
                
                for factor in factors:
                    outlook_section['content'].append(f"• {factor}")
            
            pdf_data['sections'].append(outlook_section)
            
            # Global markets section
            global_markets = report_data.get('global_markets', {})
            
            global_section = {
                'title': 'Global Markets',
                'type': 'table',
                'content': [['Region', 'Sentiment']]
            }
            
            global_section['content'].append(['US', global_markets.get('us_sentiment', 'neutral').title()])
            global_section['content'].append(['Asia', global_markets.get('asian_sentiment', 'neutral').title()])
            global_section['content'].append(['Europe', global_markets.get('european_sentiment', 'neutral').title()])
            global_section['content'].append(['Overall', global_markets.get('global_sentiment', 'neutral').title()])
            
            pdf_data['sections'].append(global_section)
            
            # Market indices section
            indices_section = {
                'title': 'Market Indices',
                'type': 'table',
                'content': [['Index', 'Last Close', 'Change']]
            }
            
            # US markets
            us_markets = global_markets.get('us_markets', {})
            
            for name, data in us_markets.items():
                close = data.get('close', 0)
                change = data.get('percent_change', 0)
                
                indices_section['content'].append([name, f"{close:.2f}", f"{change:+.2f}%"])
            
            # Asian markets
            asian_markets = global_markets.get('asian_markets', {})
            
            for name, data in asian_markets.items():
                close = data.get('close', 0)
                change = data.get('percent_change', 0)
                
                indices_section['content'].append([name, f"{close:.2f}", f"{change:+.2f}%"])
            
            pdf_data['sections'].append(indices_section)
            
            # Economic events section
            economic_events = report_data.get('economic_events', {})
            
            high_importance = economic_events.get('high_importance', [])
            
            if high_importance:
                events_section = {
                    'title': 'Economic Events',
                    'type': 'table',
                    'content': [['Time', 'Event', 'Country', 'Importance']]
                }
                
                for event in high_importance:
                    time = event.get('time', 'TBA')
                    title = event.get('title', 'Unknown')
                    country = event.get('country', '')
                    importance = event.get('importance', 'high').title()
                    
                    events_section['content'].append([time, title, country, importance])
                
                pdf_data['sections'].append(events_section)
            
            # Trading opportunities section
            opportunities = report_data.get('opportunities', {})
            
            # Bullish opportunities
            bullish = opportunities.get('bullish', [])
            
            if bullish:
                bullish_section = {
                    'title': 'Bullish Opportunities',
                    'type': 'table',
                    'content': [['Symbol', 'Confidence', 'Target', 'Stop']]
                }
                
                for opp in bullish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = f"{opp.get('confidence', 0) * 100:.1f}%"
                    target = f"₹{opp.get('target_price', 0):.2f}" if opp.get('target_price', 0) > 0 else 'N/A'
                    stop = f"₹{opp.get('stop_loss', 0):.2f}" if opp.get('stop_loss', 0) > 0 else 'N/A'
                    
                    bullish_section['content'].append([symbol, confidence, target, stop])
                
                pdf_data['sections'].append(bullish_section)
            
            # Bearish opportunities
            bearish = opportunities.get('bearish', [])
            
            if bearish:
                bearish_section = {
                    'title': 'Bearish Opportunities',
                    'type': 'table',
                    'content': [['Symbol', 'Confidence', 'Target', 'Stop']]
                }
                
                for opp in bearish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = f"{opp.get('confidence', 0) * 100:.1f}%"
                    target = f"₹{opp.get('target_price', 0):.2f}" if opp.get('target_price', 0) > 0 else 'N/A'
                    stop = f"₹{opp.get('stop_loss', 0):.2f}" if opp.get('stop_loss', 0) > 0 else 'N/A'
                    
                    bearish_section['content'].append([symbol, confidence, target, stop])
                
                pdf_data['sections'].append(bearish_section)
            
            # Important news section
            news = report_data.get('news', [])
            
            if news:
                news_section = {
                    'title': 'Important News',
                    'type': 'table',
                    'content': [['Title', 'Source', 'Sentiment']]
                }
                
                for item in news:
                    title = item.get('title', 'Unknown')
                    source = item.get('source', '')
                    sentiment = item.get('sentiment_score', 0)
                    
                    sentiment_str = "Positive" if sentiment > 0.3 else "Negative" if sentiment < -0.3 else "Neutral"
                    
                    news_section['content'].append([title, source, sentiment_str])
                
                pdf_data['sections'].append(news_section)
            
            # Footer section
            footer_section = {
                'title': 'Disclaimer',
                'type': 'text',
                'content': [
                    "This report was automatically generated by the Automated Trading System.",
                    "All analysis is based on historical data and should be used for informational purposes only.",
                    "Past performance is not indicative of future results."
                ]
            }
            
            pdf_data['sections'].append(footer_section)
            
            return pdf_data
            
        except Exception as e:
            self.logger.error(f"Error formatting morning report for PDF: {e}")
            return {"error": str(e)}
    
    def format_generic_report(self, report_data):
        """
        Format generic report for PDF.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            dict: PDF content data
        """
        try:
            # Basic formatting for generic report
            pdf_data = {
                'title': 'Report',
                'date': datetime.now().strftime("%A, %B %d, %Y"),
                'sections': []
            }
            
            # Set title if available
            if 'title' in report_data:
                pdf_data['title'] = report_data['title']
            
            # Set date if available
            if 'date' in report_data:
                pdf_data['date'] = report_data['date'].strftime("%A, %B %d, %Y")
            
            # Add error section if error occurred
            if 'error' in report_data:
                error_section = {
                    'title': 'Error',
                    'type': 'text',
                    'content': [report_data['error']]
                }
                
                pdf_data['sections'].append(error_section)
                
                return pdf_data
            
            # Format each section
            for key, value in report_data.items():
                if key in ['date', 'title', 'error']:
                    continue
                
                # Convert section to title case for header
                section_title = key.replace('_', ' ').title()
                
                if isinstance(value, dict):
                    # Dictionary section - create table
                    section = {
                        'title': section_title,
                        'type': 'table',
                        'content': [['Key', 'Value']]
                    }
                    
                    for k, v in value.items():
                        item_title = k.replace('_', ' ').title()
                        
                        if isinstance(v, (dict, list)):
                            item_value = "(complex data)"
                        else:
                            item_value = str(v)
                        
                        section['content'].append([item_title, item_value])
                    
                    pdf_data['sections'].append(section)
                    
                elif isinstance(value, list):
                    # List section
                    if value and isinstance(value[0], dict):
                        # List of dictionaries - create table
                        # Try to find common keys
                        keys = set()
                        for item in value:
                            keys.update(item.keys())
                        
                        keys = sorted(list(keys))
                        
                        if len(keys) > 0:
                            section = {
                                'title': section_title,
                                'type': 'table',
                                'content': [keys]
                            }
                            
                            for item in value:
                                row = []
                                for key in keys:
                                    row.append(str(item.get(key, '')))
                                
                                section['content'].append(row)
                            
                            pdf_data['sections'].append(section)
                        else:
                            # Empty list or no common keys
                            section = {
                                'title': section_title,
                                'type': 'text',
                                'content': ["No data available"]
                            }
                            
                            pdf_data['sections'].append(section)
                    else:
                        # Simple list - create text
                        section = {
                            'title': section_title,
                            'type': 'text',
                            'content': []
                        }
                        
                        for item in value:
                            section['content'].append(f"• {item}")
                        
                        pdf_data['sections'].append(section)
                else:
                    # Simple value - create text
                    section = {
                        'title': section_title,
                        'type': 'text',
                        'content': [str(value)]
                    }
                    
                    pdf_data['sections'].append(section)
            
            return pdf_data
            
        except Exception as e:
            self.logger.error(f"Error formatting generic report for PDF: {e}")
            return {"error": str(e)}