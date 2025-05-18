# reports/report_formatter.py (Session 46: Report Generation System)

import logging
from datetime import datetime, timedelta

class ReportFormatter:
    """
    Formats reports for different outputs.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the report formatter.
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Report formatter initialized")
    
    def format(self, report_data, format_type='markdown'):
        """
        Format report data.
        
        Args:
            report_data (dict): Report data
            format_type (str): Format type (markdown, html, json)
            
        Returns:
            str: Formatted report
        """
        try:
            self.logger.info(f"Formatting report as {format_type}")
            
            # Determine report type
            if 'market_summary' in report_data and 'prediction_summary' in report_data:
                report_type = 'daily_prediction'
            elif 'portfolio_performance' in report_data:
                report_type = 'performance'
            elif 'trades' in report_data:
                report_type = 'trading'
            else:
                report_type = 'generic'
            
            # Format based on type and format
            if format_type == 'markdown':
                if report_type == 'daily_prediction':
                    return self.format_daily_prediction_markdown(report_data)
                elif report_type == 'performance':
                    return self.format_performance_markdown(report_data)
                elif report_type == 'trading':
                    return self.format_trading_markdown(report_data)
                else:
                    return self.format_generic_markdown(report_data)
            elif format_type == 'html':
                if report_type == 'daily_prediction':
                    return self.format_daily_prediction_html(report_data)
                elif report_type == 'performance':
                    return self.format_performance_html(report_data)
                elif report_type == 'trading':
                    return self.format_trading_html(report_data)
                else:
                    return self.format_generic_html(report_data)
            elif format_type == 'json':
                import json
                return json.dumps(report_data, default=str, indent=2)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
            
        except Exception as e:
            self.logger.error(f"Error formatting report: {e}")
            return f"Error formatting report: {e}"
    
    def format_daily_prediction_markdown(self, report_data):
        """
        Format daily prediction report as Markdown.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: Markdown formatted report
        """
        try:
            date_str = report_data.get('date', datetime.now()).strftime("%A, %B %d, %Y")
            
            market_summary = report_data.get('market_summary', {})
            prediction_summary = report_data.get('prediction_summary', {})
            sector_summary = report_data.get('sector_summary', {})
            top_opportunities = report_data.get('top_opportunities', {})
            
            # Start with header
            md = f"# Daily Prediction Report - {date_str}\n\n"
            
            # Market summary
            md += "## Market Summary\n\n"
            
            direction = market_summary.get('direction', 'unknown').title()
            md += f"Today's market is **{direction}** with "
            
            advances = market_summary.get('advances', 0)
            declines = market_summary.get('declines', 0)
            
            md += f"{advances} advances and {declines} declines.\n\n"
            
            # Market indices
            md += "### Market Indices\n\n"
            
            indices = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
            for index in indices:
                index_data = market_summary.get(index.lower(), {})
                change = index_data.get('change', 0)
                
                if change > 0:
                    md += f"* **{index}**: {index_data.get('close', 0):.2f} (▲ +{change:.2f}%)\n"
                else:
                    md += f"* **{index}**: {index_data.get('close', 0):.2f} (▼ {change:.2f}%)\n"
            
            md += "\n"
            
            # Prediction summary
            md += "## Prediction Summary\n\n"
            
            count = prediction_summary.get('count', 0)
            up_count = prediction_summary.get('up_count', 0)
            down_count = prediction_summary.get('down_count', 0)
            
            md += f"Today we have **{count} predictions** ({up_count} bullish, {down_count} bearish).\n\n"
            
            yesterday_accuracy = prediction_summary.get('yesterday_accuracy', 0)
            md += f"Yesterday's prediction accuracy: **{yesterday_accuracy:.2f}%**\n\n"
            
            # Sector summary
            md += "## Sector Summary\n\n"
            
            rotation_pattern = sector_summary.get('rotation_pattern', 'unknown')
            md += f"Sector rotation pattern: **{rotation_pattern}**\n\n"
            
            # Top performing sectors
            top_sectors = sector_summary.get('top_sectors', [])
            
            if top_sectors:
                md += "### Top Performing Sectors\n\n"
                
                for sector in top_sectors:
                    name = sector.get('name', 'Unknown')
                    change = sector.get('change', 0)
                    
                    md += f"* **{name}**: +{change:.2f}%\n"
                
                md += "\n"
            
            # Bottom performing sectors
            bottom_sectors = sector_summary.get('bottom_sectors', [])
            
            if bottom_sectors:
                md += "### Underperforming Sectors\n\n"
                
                for sector in bottom_sectors:
                    name = sector.get('name', 'Unknown')
                    change = sector.get('change', 0)
                    
                    md += f"* **{name}**: {change:.2f}%\n"
                
                md += "\n"
            
            # Top opportunities
            md += "## Top Trading Opportunities\n\n"
            
            # Bullish opportunities
            bullish = top_opportunities.get('bullish', [])
            
            if bullish:
                md += "### Bullish Opportunities\n\n"
                
                for opp in bullish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = opp.get('confidence', 0) * 100
                    target = opp.get('target_price', 0)
                    stop = opp.get('stop_loss', 0)
                    
                    md += f"* **{symbol}** ({confidence:.1f}% confidence)"
                    
                    if target > 0:
                        md += f", Target: ₹{target:.2f}"
                    
                    if stop > 0:
                        md += f", Stop: ₹{stop:.2f}"
                    
                    md += "\n"
                
                md += "\n"
            
            # Bearish opportunities
            bearish = top_opportunities.get('bearish', [])
            
            if bearish:
                md += "### Bearish Opportunities\n\n"
                
                for opp in bearish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = opp.get('confidence', 0) * 100
                    target = opp.get('target_price', 0)
                    stop = opp.get('stop_loss', 0)
                    
                    md += f"* **{symbol}** ({confidence:.1f}% confidence)"
                    
                    if target > 0:
                        md += f", Target: ₹{target:.2f}"
                    
                    if stop > 0:
                        md += f", Stop: ₹{stop:.2f}"
                    
                    md += "\n"
                
                md += "\n"
            
            # Footer
            md += "---\n"
            md += "*This report was automatically generated by the Automated Trading System.*\n"
            
            return md
            
        except Exception as e:
            self.logger.error(f"Error formatting daily prediction as Markdown: {e}")
            return f"Error formatting report: {e}"
    
    def format_daily_prediction_html(self, report_data):
        """
        Format daily prediction report as HTML.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: HTML formatted report
        """
        try:
            date_str = report_data.get('date', datetime.now()).strftime("%A, %B %d, %Y")
            
            market_summary = report_data.get('market_summary', {})
            prediction_summary = report_data.get('prediction_summary', {})
            sector_summary = report_data.get('sector_summary', {})
            top_opportunities = report_data.get('top_opportunities', {})
            
            # Start with HTML template
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Daily Prediction Report - {date_str}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333366; }}
                    h2 {{ color: #335588; margin-top: 20px; border-bottom: 1px solid #ccc; }}
                    h3 {{ color: #337799; }}
                    .up {{ color: green; }}
                    .down {{ color: red; }}
                    .neutral {{ color: gray; }}
                    .section {{ margin-bottom: 20px; }}
                    .footer {{ margin-top: 40px; font-size: 12px; color: #777; }}
                </style>
            </head>
            <body>
                <h1>Daily Prediction Report - {date_str}</h1>
            """
            
            # Market summary
            html += """
                <div class="section">
                    <h2>Market Summary</h2>
            """
            
            direction = market_summary.get('direction', 'unknown').title()
            advances = market_summary.get('advances', 0)
            declines = market_summary.get('declines', 0)
            
            # Set direction class
            direction_class = ""
            if "positive" in direction.lower():
                direction_class = "up"
            elif "negative" in direction.lower():
                direction_class = "down"
            else:
                direction_class = "neutral"
            
            html += f"""
                    <p>Today's market is <strong class="{direction_class}">{direction}</strong> with 
                    {advances} advances and {declines} declines.</p>
                    
                    <h3>Market Indices</h3>
                    <ul>
            """
            
            # Market indices
            indices = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
            for index in indices:
                index_data = market_summary.get(index.lower(), {})
                change = index_data.get('change', 0)
                
                if change > 0:
                    html += f"""
                        <li><strong>{index}</strong>: {index_data.get('close', 0):.2f} (<span class="up">▲ +{change:.2f}%</span>)</li>
                    """
                else:
                    html += f"""
                        <li><strong>{index}</strong>: {index_data.get('close', 0):.2f} (<span class="down">▼ {change:.2f}%</span>)</li>
                    """
            
            html += """
                    </ul>
                </div>
            """
            
            # Prediction summary
            html += """
                <div class="section">
                    <h2>Prediction Summary</h2>
            """
            
            count = prediction_summary.get('count', 0)
            up_count = prediction_summary.get('up_count', 0)
            down_count = prediction_summary.get('down_count', 0)
            
            html += f"""
                    <p>Today we have <strong>{count} predictions</strong> ({up_count} bullish, {down_count} bearish).</p>
            """
            
            yesterday_accuracy = prediction_summary.get('yesterday_accuracy', 0)
            html += f"""
                    <p>Yesterday's prediction accuracy: <strong>{yesterday_accuracy:.2f}%</strong></p>
                </div>
            """
            
            # Sector summary
            html += """
                <div class="section">
                    <h2>Sector Summary</h2>
            """
            
            rotation_pattern = sector_summary.get('rotation_pattern', 'unknown')
            html += f"""
                    <p>Sector rotation pattern: <strong>{rotation_pattern}</strong></p>
            """
            
            # Top performing sectors
            top_sectors = sector_summary.get('top_sectors', [])
            
            if top_sectors:
                html += """
                    <h3>Top Performing Sectors</h3>
                    <ul>
                """
                
                for sector in top_sectors:
                    name = sector.get('name', 'Unknown')
                    change = sector.get('change', 0)
                    
                    html += f"""
                        <li><strong>{name}</strong>: <span class="up">+{change:.2f}%</span></li>
                    """
                
                html += """
                    </ul>
                """
            
            # Bottom performing sectors
            bottom_sectors = sector_summary.get('bottom_sectors', [])
            
            if bottom_sectors:
                html += """
                    <h3>Underperforming Sectors</h3>
                    <ul>
                """
                
                for sector in bottom_sectors:
                    name = sector.get('name', 'Unknown')
                    change = sector.get('change', 0)
                    
                    html += f"""
                        <li><strong>{name}</strong>: <span class="down">{change:.2f}%</span></li>
                    """
                
                html += """
                    </ul>
                """
            
            html += """
                </div>
            """
            
            # Top opportunities
            html += """
                <div class="section">
                    <h2>Top Trading Opportunities</h2>
            """
            
            # Bullish opportunities
            bullish = top_opportunities.get('bullish', [])
            
            if bullish:
                html += """
                    <h3>Bullish Opportunities</h3>
                    <ul>
                """
                
                for opp in bullish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = opp.get('confidence', 0) * 100
                    target = opp.get('target_price', 0)
                    stop = opp.get('stop_loss', 0)
                    
                    html += f"""
                        <li><strong>{symbol}</strong> ({confidence:.1f}% confidence)
                    """
                    
                    if target > 0:
                        html += f", Target: ₹{target:.2f}"
                    
                    if stop > 0:
                        html += f", Stop: ₹{stop:.2f}"
                    
                    html += "</li>"
                
                html += """
                    </ul>
                """
            
            # Bearish opportunities
            bearish = top_opportunities.get('bearish', [])
            
            if bearish:
                html += """
                    <h3>Bearish Opportunities</h3>
                    <ul>
                """
                
                for opp in bearish:
                    symbol = opp.get('symbol', 'Unknown')
                    confidence = opp.get('confidence', 0) * 100
                    target = opp.get('target_price', 0)
                    stop = opp.get('stop_loss', 0)
                    
                    html += f"""
                        <li><strong>{symbol}</strong> ({confidence:.1f}% confidence)
                    """
                    
                    if target > 0:
                        html += f", Target: ₹{target:.2f}"
                    
                    if stop > 0:
                        html += f", Stop: ₹{stop:.2f}"
                    
                    html += "</li>"
                
                html += """
                    </ul>
                """
            
            html += """
                </div>
            """
            
            # Footer
            html += """
                <div class="footer">
                    <hr>
                    <p>This report was automatically generated by the Automated Trading System.</p>
                </div>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error formatting daily prediction as HTML: {e}")
            return f"<html><body><h1>Error</h1><p>{e}</p></body></html>"
    
    def format_performance_markdown(self, report_data):
        """
        Format performance report as Markdown.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: Markdown formatted report
        """
        # Implementation for performance report formatting in Markdown
        # Similar structure as daily prediction but with performance metrics
        # Left as an exercise or for future implementation
        return "Performance Report (Markdown format not implemented yet)"
    
    def format_performance_html(self, report_data):
        """
        Format performance report as HTML.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: HTML formatted report
        """
        # Implementation for performance report formatting in HTML
        # Similar structure as daily prediction but with performance metrics
        # Left as an exercise or for future implementation
        return "<html><body><h1>Performance Report</h1><p>HTML format not implemented yet</p></body></html>"
    
    def format_trading_markdown(self, report_data):
        """
        Format trading report as Markdown.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: Markdown formatted report
        """
        # Implementation for trading report formatting in Markdown
        # Similar structure as daily prediction but with trading metrics
        # Left as an exercise or for future implementation
        return "Trading Report (Markdown format not implemented yet)"
    
    def format_trading_html(self, report_data):
        """
        Format trading report as HTML.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: HTML formatted report
        """
        # Implementation for trading report formatting in HTML
        # Similar structure as daily prediction but with trading metrics
        # Left as an exercise or for future implementation
        return "<html><body><h1>Trading Report</h1><p>HTML format not implemented yet</p></body></html>"
    
    def format_generic_markdown(self, report_data):
        """
        Format generic report as Markdown.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: Markdown formatted report
        """
        try:
            # Basic formatting for generic report
            md = "# Generic Report\n\n"
            
            # Format date if available
            if 'date' in report_data:
                date_str = report_data['date'].strftime("%A, %B %d, %Y")
                md += f"Generated on: {date_str}\n\n"
            
            # Format each section
            for key, value in report_data.items():
                if key == 'date':
                    continue
                
                # Convert section to title case for header
                section = key.replace('_', ' ').title()
                md += f"## {section}\n\n"
                
                # Format section content
                if isinstance(value, dict):
                    for k, v in value.items():
                        item = k.replace('_', ' ').title()
                        md += f"**{item}**: {v}\n"
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                sub_item = k.replace('_', ' ').title()
                                md += f"* **{sub_item}**: {v}\n"
                        else:
                            md += f"* {item}\n"
                else:
                    md += f"{value}\n"
                
                md += "\n"
            
            # Footer
            md += "---\n"
            md += "*This report was automatically generated.*\n"
            
            return md
            
        except Exception as e:
            self.logger.error(f"Error formatting generic report as Markdown: {e}")
            return f"Error formatting report: {e}"
    
    def format_generic_html(self, report_data):
        """
        Format generic report as HTML.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: HTML formatted report
        """
        try:
            # Basic HTML template
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Generic Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333366; }
                    h2 { color: #335588; margin-top: 20px; border-bottom: 1px solid #ccc; }
                    .section { margin-bottom: 20px; }
                    .footer { margin-top: 40px; font-size: 12px; color: #777; }
                </style>
            </head>
            <body>
                <h1>Generic Report</h1>
            """
            
            # Format date if available
            if 'date' in report_data:
                date_str = report_data['date'].strftime("%A, %B %d, %Y")
                html += f"""
                <p>Generated on: {date_str}</p>
                """
            
            # Format each section
            for key, value in report_data.items():
                if key == 'date':
                    continue
                
                # Convert section to title case for header
                section = key.replace('_', ' ').title()
                html += f"""
                <div class="section">
                    <h2>{section}</h2>
                """
                
                # Format section content
                if isinstance(value, dict):
                    html += "<ul>"
                    for k, v in value.items():
                        item = k.replace('_', ' ').title()
                        html += f"<li><strong>{item}</strong>: {v}</li>"
                    html += "</ul>"
                elif isinstance(value, list):
                    html += "<ul>"
                    for item in value:
                        if isinstance(item, dict):
                            html += "<li>"
                            for k, v in item.items():
                                sub_item = k.replace('_', ' ').title()
                                html += f"<strong>{sub_item}</strong>: {v}<br>"
                            html += "</li>"
                        else:
                            html += f"<li>{item}</li>"
                    html += "</ul>"
                else:
                    html += f"<p>{value}</p>"
                
                html += """
                </div>
                """
            
            # Footer
            html += """
                <div class="footer">
                    <hr>
                    <p>This report was automatically generated.</p>
                </div>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error formatting generic report as HTML: {e}")
            return f"<html><body><h1>Error</h1><p>{e}</p></body></html>"