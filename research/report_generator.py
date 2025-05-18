"""
Research Report Generator

This module generates comprehensive research reports based on the analysis
from various components of the automated trading system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import math
import json
import os
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import base64

class ReportGenerator:
    """
    Generates research reports for different purposes:
    - Individual stock analysis
    - Market overview
    - Portfolio performance
    - Trading opportunities
    - Correlation and risk analysis
    """
    
    def __init__(self, db_connector):
        """
        Initialize the report generator with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Try to import various analyzers for comprehensive reports
        self.volatility_analyzer = None
        self.correlation_analyzer = None
        self.opportunity_scanner = None
        self.technical_analyzer = None
        self.fundamental_analyzer = None
        
        try:
            from research.volatility_analyzer import VolatilityAnalyzer
            self.volatility_analyzer = VolatilityAnalyzer(db_connector)
        except ImportError:
            self.logger.warning("VolatilityAnalyzer not found. Report functionality will be limited.")
        
        try:
            from research.correlation_analyzer import CorrelationAnalyzer
            self.correlation_analyzer = CorrelationAnalyzer(db_connector)
        except ImportError:
            self.logger.warning("CorrelationAnalyzer not found. Report functionality will be limited.")
        
        try:
            from research.opportunity_scanner import OpportunityScanner
            self.opportunity_scanner = OpportunityScanner(db_connector)
        except ImportError:
            self.logger.warning("OpportunityScanner not found. Report functionality will be limited.")
        
        try:
            from research.technical_analyzer import TechnicalAnalyzer
            self.technical_analyzer = TechnicalAnalyzer(db_connector)
        except ImportError:
            self.logger.warning("TechnicalAnalyzer not found. Report functionality will be limited.")
        
        try:
            from research.fundamental_analyzer import FundamentalAnalyzer
            self.fundamental_analyzer = FundamentalAnalyzer(db_connector)
        except ImportError:
            self.logger.warning("FundamentalAnalyzer not found. Report functionality will be limited.")
        
        # Define report configurations
        self.report_configs = {
            "stock_analysis": {
                "sections": [
                    "overview",
                    "technical_analysis",
                    "fundamental_analysis",
                    "volatility_analysis",
                    "price_levels",
                    "opportunities",
                    "correlation_analysis",
                    "summary_and_recommendation"
                ]
            },
            "market_overview": {
                "sections": [
                    "market_summary",
                    "index_performance",
                    "sector_performance",
                    "market_breadth",
                    "volatility_overview",
                    "correlation_overview",
                    "sentiment_analysis",
                    "upcoming_events"
                ]
            },
            "portfolio_analysis": {
                "sections": [
                    "portfolio_summary",
                    "performance_metrics",
                    "position_analysis",
                    "risk_metrics",
                    "correlation_matrix",
                    "sector_allocation",
                    "optimization_suggestions"
                ]
            },
            "opportunity_report": {
                "sections": [
                    "summary",
                    "breakout_opportunities",
                    "trend_following_opportunities",
                    "mean_reversion_opportunities",
                    "volatility_opportunities",
                    "pair_trading_opportunities",
                    "support_resistance_opportunities"
                ]
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for this module."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def generate_stock_report(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Generate a comprehensive stock analysis report.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with report data and content
        """
        try:
            self.logger.info(f"Generating stock report for {symbol} ({exchange})")
            
            # Start with basic report structure
            report = {
                "type": "stock_analysis",
                "symbol": symbol,
                "exchange": exchange,
                "generated_at": datetime.now(),
                "sections": {}
            }
            
            # Get market data
            data = self._get_market_data(symbol, exchange)
            
            if not data or len(data) < 30:
                return {
                    "status": "error",
                    "error": "Insufficient market data"
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # 1. Overview Section
            report["sections"]["overview"] = self._generate_overview_section(symbol, exchange, df)
            
            # 2. Technical Analysis Section
            if self.technical_analyzer:
                tech_analysis = self.technical_analyzer.analyze(symbol, exchange)
                report["sections"]["technical_analysis"] = self._format_technical_analysis(tech_analysis)
            else:
                # Simplified technical analysis if analyzer not available
                report["sections"]["technical_analysis"] = self._generate_simple_technical_analysis(df)
            
            # 3. Fundamental Analysis Section
            if self.fundamental_analyzer:
                fund_analysis = self.fundamental_analyzer.analyze(symbol, exchange)
                report["sections"]["fundamental_analysis"] = self._format_fundamental_analysis(fund_analysis)
            else:
                # Get any available financial data from database
                report["sections"]["fundamental_analysis"] = self._get_financial_data(symbol, exchange)
            
            # 4. Volatility Analysis Section
            if self.volatility_analyzer:
                vol_analysis = self.volatility_analyzer.analyze_volatility(symbol, exchange)
                report["sections"]["volatility_analysis"] = vol_analysis
            else:
                # Simplified volatility analysis
                report["sections"]["volatility_analysis"] = self._generate_simple_volatility_analysis(df)
            
            # 5. Price Levels Section
            report["sections"]["price_levels"] = self._generate_price_levels_section(df)
            
            # 6. Trading Opportunities Section
            if self.opportunity_scanner:
                opp_details = self.opportunity_scanner.get_opportunity_details(symbol, exchange)
                report["sections"]["opportunities"] = opp_details
            else:
                # Simplified opportunity scanning
                report["sections"]["opportunities"] = self._generate_simple_opportunities(df)
            
            # 7. Correlation Analysis Section
            if self.correlation_analyzer:
                # Get sector for this symbol
                instrument = self.db.portfolio_collection.find_one({
                    "symbol": symbol,
                    "exchange": exchange
                })
                
                if instrument and "sector" in instrument:
                    sector = instrument["sector"]
                    # Get symbols in same sector
                    sector_symbols = self._get_sector_symbols(sector, exchange)
                    
                    if len(sector_symbols) >= 3:
                        corr_analysis = self.correlation_analyzer.analyze_correlation_matrix(
                            sector_symbols[:10],  # Top 10 symbols in sector
                            exchange
                        )
                        report["sections"]["correlation_analysis"] = self._format_correlation_analysis(
                            corr_analysis, symbol
                        )
                    else:
                        report["sections"]["correlation_analysis"] = {
                            "status": "limited_data",
                            "message": "Insufficient sector data for correlation analysis"
                        }
                else:
                    # Try correlation with index and major stocks
                    default_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "TCS", "INFY", symbol]
                    corr_analysis = self.correlation_analyzer.analyze_correlation_matrix(
                        default_symbols,
                        exchange
                    )
                    report["sections"]["correlation_analysis"] = self._format_correlation_analysis(
                        corr_analysis, symbol
                    )
            else:
                report["sections"]["correlation_analysis"] = {
                    "status": "not_available",
                    "message": "Correlation analyzer not available"
                }
            
            # 8. Summary and Recommendation Section
            report["sections"]["summary_and_recommendation"] = self._generate_summary_section(
                symbol, exchange, report["sections"]
            )
            
            # Generate charts
            report["charts"] = self._generate_stock_charts(df, symbol)
            
            # Generate formatted report content
            report["content"] = self._format_stock_report(report)
            
            # Save report to database
            self._save_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating stock report for {symbol}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_market_report(self, indices: List[str] = None, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Generate a comprehensive market overview report.
        
        Args:
            indices: List of indices to analyze
            exchange: Stock exchange
            
        Returns:
            Dictionary with report data and content
        """
        try:
            self.logger.info(f"Generating market overview report")
            
            # Use default indices if none provided
            if not indices:
                indices = ["NIFTY", "BANKNIFTY", "NIFTY IT", "NIFTY PHARMA", "NIFTY AUTO"]
            
            # Start with basic report structure
            report = {
                "type": "market_overview",
                "exchange": exchange,
                "generated_at": datetime.now(),
                "sections": {}
            }
            
            # 1. Market Summary Section
            report["sections"]["market_summary"] = self._generate_market_summary(indices, exchange)
            
            # 2. Index Performance Section
            report["sections"]["index_performance"] = self._generate_index_performance(indices, exchange)
            
            # 3. Sector Performance Section
            report["sections"]["sector_performance"] = self._generate_sector_performance(exchange)
            
            # 4. Market Breadth Section
            report["sections"]["market_breadth"] = self._generate_market_breadth(exchange)
            
            # 5. Volatility Overview Section
            report["sections"]["volatility_overview"] = self._generate_volatility_overview(indices, exchange)
            
            # 6. Correlation Overview Section
            report["sections"]["correlation_overview"] = self._generate_correlation_overview(indices, exchange)
            
            # 7. Sentiment Analysis Section
            report["sections"]["sentiment_analysis"] = self._generate_sentiment_analysis(exchange)
            
            # 8. Upcoming Events Section
            report["sections"]["upcoming_events"] = self._get_upcoming_events(exchange)
            
            # Generate formatted report content
            report["content"] = self._format_market_report(report)
            
            # Generate charts
            report["charts"] = self._generate_market_charts(indices, exchange)
            
            # Save report to database
            self._save_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating market report: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_portfolio_report(self, portfolio_id: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive portfolio analysis report.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dictionary with report data and content
        """
        try:
            self.logger.info(f"Generating portfolio report for {portfolio_id}")
            
            # Get portfolio data
            portfolio = self._get_portfolio_data(portfolio_id)
            
            if not portfolio or "positions" not in portfolio or not portfolio["positions"]:
                return {
                    "status": "error",
                    "error": "Portfolio not found or empty"
                }
            
            # Start with basic report structure
            report = {
                "type": "portfolio_analysis",
                "portfolio_id": portfolio_id,
                "generated_at": datetime.now(),
                "sections": {}
            }
            
            # 1. Portfolio Summary Section
            report["sections"]["portfolio_summary"] = self._generate_portfolio_summary(portfolio)
            
            # 2. Performance Metrics Section
            report["sections"]["performance_metrics"] = self._generate_performance_metrics(portfolio)
            
            # 3. Position Analysis Section
            report["sections"]["position_analysis"] = self._generate_position_analysis(portfolio)
            
            # 4. Risk Metrics Section
            report["sections"]["risk_metrics"] = self._generate_risk_metrics(portfolio)
            
            # 5. Correlation Matrix Section
            report["sections"]["correlation_matrix"] = self._generate_portfolio_correlation(portfolio)
            
            # 6. Sector Allocation Section
            report["sections"]["sector_allocation"] = self._generate_sector_allocation(portfolio)
            
            # 7. Optimization Suggestions Section
            report["sections"]["optimization_suggestions"] = self._generate_optimization_suggestions(
                portfolio, report["sections"]
            )
            
            # Generate formatted report content
            report["content"] = self._format_portfolio_report(report)
            
            # Generate charts
            report["charts"] = self._generate_portfolio_charts(portfolio)
            
            # Save report to database
            self._save_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio report: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_opportunity_report(self, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Generate a comprehensive trading opportunities report.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            Dictionary with report data and content
        """
        try:
            self.logger.info(f"Generating trading opportunities report")
            
            # Start with basic report structure
            report = {
                "type": "opportunity_report",
                "exchange": exchange,
                "generated_at": datetime.now(),
                "sections": {}
            }
            
            # Scan for opportunities if scanner is available
            if self.opportunity_scanner:
                opportunities = self.opportunity_scanner.scan_all_opportunities(exchange)
                report["sections"] = opportunities
            else:
                # Get recent opportunities from database
                recent_scan = self.db.opportunity_scan_collection.find_one(
                    {}, sort=[("timestamp", -1)]
                )
                
                if recent_scan:
                    report["sections"] = recent_scan
                else:
                    report["sections"] = {
                        "status": "not_available",
                        "message": "No recent opportunity scans found"
                    }
            
            # Generate formatted report content
            report["content"] = self._format_opportunity_report(report)
            
            # Save report to database
            self._save_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating opportunity report: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_overview_section(self, symbol: str, exchange: str, 
                               df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate overview section for stock report.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            df: DataFrame with market data
            
        Returns:
            Dictionary with overview data
        """
        # Get instrument details from database
        instrument = self.db.portfolio_collection.find_one({
            "symbol": symbol,
            "exchange": exchange
        })
        
        # Basic price metrics
        latest_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2] if len(df) > 1 else None
        day_change = (latest_close / prev_close - 1) * 100 if prev_close else None
        
        week_ago = df['close'].iloc[-6] if len(df) > 5 else None
        week_change = (latest_close / week_ago - 1) * 100 if week_ago else None
        
        month_ago = df['close'].iloc[-22] if len(df) > 21 else None
        month_change = (latest_close / month_ago - 1) * 100 if month_ago else None
        
        year_ago = df['close'].iloc[-252] if len(df) > 251 else None
        year_change = (latest_close / year_ago - 1) * 100 if year_ago else None
        
        # Trading range
        high_52w = df['high'].rolling(window=252).max().iloc[-1] if len(df) > 251 else df['high'].max()
        low_52w = df['low'].rolling(window=252).min().iloc[-1] if len(df) > 251 else df['low'].min()
        
        # Proximity to 52-week range
        range_size = high_52w - low_52w
        if range_size > 0:
            range_percentile = (latest_close - low_52w) / range_size * 100
        else:
            range_percentile = 50
        
        # Moving Averages
        ma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
        ma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        ma_200 = df['close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
        
        # Volume metrics
        if 'volume' in df.columns:
            avg_volume = df['volume'].iloc[-20:].mean()
            latest_volume = df['volume'].iloc[-1]
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1
        else:
            avg_volume = None
            latest_volume = None
            volume_ratio = None
        
        # Company information
        company_info = {}
        if instrument:
            company_info = {
                "sector": instrument.get("sector", "Unknown"),
                "industry": instrument.get("industry", "Unknown"),
                "added_date": instrument.get("added_date", None)
            }
        
        # Market cap if available
        market_cap = None
        financial_data = self.db.financial_data_collection.find_one(
            {"symbol": symbol, "exchange": exchange},
            sort=[("timestamp", -1)]
        )
        
        if financial_data and "market_cap" in financial_data:
            market_cap = financial_data["market_cap"]
        
        # Create overview section
        overview = {
            "symbol": symbol,
            "exchange": exchange,
            "current_price": latest_close,
            "price_changes": {
                "daily": day_change,
                "weekly": week_change,
                "monthly": month_change,
                "yearly": year_change
            },
            "price_range": {
                "high_52w": high_52w,
                "low_52w": low_52w,
                "range_percentile": range_percentile
            },
            "moving_averages": {
                "ma_20": ma_20,
                "ma_50": ma_50,
                "ma_200": ma_200,
                "vs_ma20": ((latest_close / ma_20) - 1) * 100 if ma_20 else None,
                "vs_ma50": ((latest_close / ma_50) - 1) * 100 if ma_50 else None,
                "vs_ma200": ((latest_close / ma_200) - 1) * 100 if ma_200 else None
            },
            "volume": {
                "latest": latest_volume,
                "average": avg_volume,
                "ratio": volume_ratio
            },
            "company_info": company_info,
            "market_cap": market_cap
        }
        
        # Generate summary text
        summary_text = self._generate_overview_summary(overview)
        overview["summary"] = summary_text
        
        return overview
    
    def _generate_overview_summary(self, overview: Dict[str, Any]) -> str:
        """
        Generate summary text for overview section.
        
        Args:
            overview: Overview data dictionary
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        symbol = overview["symbol"]
        price = overview["current_price"]
        
        # Basic company info
        company_info = overview["company_info"]
        sector = company_info.get("sector", "Unknown")
        industry = company_info.get("industry", "Unknown")
        
        summary_parts.append(f"{symbol} is currently trading at ₹{price:.2f}.")
        
        if sector != "Unknown":
            summary_parts.append(f"It operates in the {sector} sector, specifically in the {industry} industry.")
        
        # Price changes
        price_changes = overview["price_changes"]
        
        day_change = price_changes.get("daily")
        if day_change is not None:
            change_str = f"up {day_change:.2f}%" if day_change > 0 else f"down {abs(day_change):.2f}%"
            summary_parts.append(f"The stock is {change_str} today.")
        
        month_change = price_changes.get("monthly")
        if month_change is not None:
            change_str = f"up {month_change:.2f}%" if month_change > 0 else f"down {abs(month_change):.2f}%"
            summary_parts.append(f"Over the past month, it has moved {change_str}.")
        
        year_change = price_changes.get("yearly")
        if year_change is not None:
            change_str = f"up {year_change:.2f}%" if year_change > 0 else f"down {abs(year_change):.2f}%"
            summary_parts.append(f"The yearly performance shows a {change_str} move.")
        
        # 52-week range
        price_range = overview["price_range"]
        high_52w = price_range.get("high_52w")
        low_52w = price_range.get("low_52w")
        range_percentile = price_range.get("range_percentile")
        
        if high_52w is not None and low_52w is not None:
            summary_parts.append(f"The 52-week range is ₹{low_52w:.2f} to ₹{high_52w:.2f}.")
            
            if range_percentile is not None:
                if range_percentile > 80:
                    summary_parts.append(f"The current price is near the upper end of its 52-week range.")
                elif range_percentile < 20:
                    summary_parts.append(f"The current price is near the lower end of its 52-week range.")
                else:
                    summary_parts.append(f"The current price is in the middle of its 52-week range.")
        
        # Moving averages
        moving_averages = overview["moving_averages"]
        ma_50 = moving_averages.get("ma_50")
        vs_ma50 = moving_averages.get("vs_ma50")
        ma_200 = moving_averages.get("ma_200")
        vs_ma200 = moving_averages.get("vs_ma200")
        
        if ma_50 is not None and vs_ma50 is not None:
            if vs_ma50 > 0:
                summary_parts.append(f"The stock is trading {vs_ma50:.2f}% above its 50-day moving average.")
            else:
                summary_parts.append(f"The stock is trading {abs(vs_ma50):.2f}% below its 50-day moving average.")
        
        if ma_200 is not None and vs_ma200 is not None:
            if vs_ma200 > 0:
                summary_parts.append(f"The stock is trading {vs_ma200:.2f}% above its 200-day moving average, indicating a long-term uptrend.")
            else:
                summary_parts.append(f"The stock is trading {abs(vs_ma200):.2f}% below its 200-day moving average, indicating a long-term downtrend.")
        
        # Volume
        volume = overview["volume"]
        volume_ratio = volume.get("ratio")
        
        if volume_ratio is not None:
            if volume_ratio > 1.5:
                summary_parts.append(f"Trading volume is {volume_ratio:.1f}x higher than the 20-day average, showing increased interest.")
            elif volume_ratio < 0.5:
                summary_parts.append(f"Trading volume is {volume_ratio:.1f}x lower than the 20-day average, showing decreased interest.")
        
        # Market cap
        market_cap = overview.get("market_cap")
        if market_cap is not None:
            if market_cap < 1000000000:  # Less than 1B
                cap_str = f"₹{market_cap/10000000:.2f} Cr"
                summary_parts.append(f"With a market capitalization of {cap_str}, it is a small-cap stock.")
            elif market_cap < 10000000000:  # Less than 10B
                cap_str = f"₹{market_cap/10000000:.2f} Cr"
                summary_parts.append(f"With a market capitalization of {cap_str}, it is a mid-cap stock.")
            else:
                cap_str = f"₹{market_cap/10000000:.2f} Cr"
                summary_parts.append(f"With a market capitalization of {cap_str}, it is a large-cap stock.")
        
        return " ".join(summary_parts)
    
    def _format_technical_analysis(self, tech_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format technical analysis data for the report.
        
        Args:
            tech_analysis: Technical analysis data
            
        Returns:
            Formatted technical analysis section
        """
        if not tech_analysis or "status" in tech_analysis and tech_analysis["status"] == "error":
            return {
                "status": "error",
                "message": tech_analysis.get("error", "Unknown error in technical analysis")
            }
        
        # Extract relevant sections
        formatted = {
            "indicators": tech_analysis.get("indicators", {}),
            "patterns": tech_analysis.get("patterns", {}),
            "trend_analysis": tech_analysis.get("trend_analysis", {}),
            "support_resistance": tech_analysis.get("support_resistance", {}),
            "summary": tech_analysis.get("summary", "")
        }
        
        return formatted
    
    def _generate_simple_technical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate simplified technical analysis when analyzer is not available.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with technical analysis
        """
        try:
            # Calculate basic indicators
            
            # Moving Averages
            df['ma_20'] = df['close'].rolling(window=20).mean()
            df['ma_50'] = df['close'].rolling(window=50).mean()
            df['ma_200'] = df['close'].rolling(window=200).mean()
            
            # MACD
            df['ema_12'] = df['close'].rolling(window=12).mean()  # Simple proxy for EMA
            df['ema_26'] = df['close'].rolling(window=26).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].rolling(window=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Determine trend
            long_trend = "bullish" if latest['close'] > latest['ma_200'] else "bearish"
            medium_trend = "bullish" if latest['close'] > latest['ma_50'] else "bearish"
            short_trend = "bullish" if latest['close'] > latest['ma_20'] else "bearish"
            
            if long_trend == medium_trend == short_trend:
                trend_strength = "strong"
            elif long_trend == medium_trend or medium_trend == short_trend:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
            
            # Check MA crossovers
            ma_20_50_cross = False
            ma_50_200_cross = False
            
            if len(df) > 2:
                prev = df.iloc[-2]
                
                if (latest['ma_20'] > latest['ma_50'] and prev['ma_20'] <= prev['ma_50']) or \
                   (latest['ma_20'] < latest['ma_50'] and prev['ma_20'] >= prev['ma_50']):
                    ma_20_50_cross = True
                
                if (latest['ma_50'] > latest['ma_200'] and prev['ma_50'] <= prev['ma_200']) or \
                   (latest['ma_50'] < latest['ma_200'] and prev['ma_50'] >= prev['ma_200']):
                    ma_50_200_cross = True
            
            # Collect indicator signals
            indicator_signals = []
            
            # MA signals
            if latest['close'] > latest['ma_20'] and latest['close'] > latest['ma_50'] and latest['close'] > latest['ma_200']:
                indicator_signals.append("Price above all major moving averages (bullish)")
            elif latest['close'] < latest['ma_20'] and latest['close'] < latest['ma_50'] and latest['close'] < latest['ma_200']:
                indicator_signals.append("Price below all major moving averages (bearish)")
            
            if latest['ma_20'] > latest['ma_50'] and latest['ma_50'] > latest['ma_200']:
                indicator_signals.append("Moving averages in bullish alignment")
            elif latest['ma_20'] < latest['ma_50'] and latest['ma_50'] < latest['ma_200']:
                indicator_signals.append("Moving averages in bearish alignment")
            
            # MACD signals
            if latest['macd'] > latest['macd_signal']:
                indicator_signals.append("MACD above signal line (bullish)")
            else:
                indicator_signals.append("MACD below signal line (bearish)")
            
            if latest['macd'] > 0:
                indicator_signals.append("MACD positive (bullish)")
            else:
                indicator_signals.append("MACD negative (bearish)")
            
            # RSI signals
            if latest['rsi'] < 30:
                indicator_signals.append("RSI in oversold territory (potential bullish reversal)")
            elif latest['rsi'] > 70:
                indicator_signals.append("RSI in overbought territory (potential bearish reversal)")
            else:
                if latest['rsi'] > 50:
                    indicator_signals.append("RSI above 50 (bullish)")
                else:
                    indicator_signals.append("RSI below 50 (bearish)")
            
            # Bollinger Band signals
            if latest['close'] < latest['bb_lower']:
                indicator_signals.append("Price below lower Bollinger Band (potential bullish reversal)")
            elif latest['close'] > latest['bb_upper']:
                indicator_signals.append("Price above upper Bollinger Band (potential bearish reversal)")
            
            # Count bullish and bearish signals
            bullish_count = sum(1 for signal in indicator_signals if "bullish" in signal.lower())
            bearish_count = sum(1 for signal in indicator_signals if "bearish" in signal.lower())
            
            # Overall bias
            if bullish_count > bearish_count:
                overall_bias = "bullish"
            elif bearish_count > bullish_count:
                overall_bias = "bearish"
            else:
                overall_bias = "neutral"
            
            # Find support and resistance levels (simplified)
            support_resistance = self._find_simple_support_resistance(df)
            
            # Generate summary text
            summary = f"Technical analysis shows a {overall_bias} bias with {bullish_count} bullish signals and {bearish_count} bearish signals. "
            summary += f"The stock is in a {trend_strength} {long_term} long-term trend. "
            
            if ma_20_50_cross:
                cross_type = "bullish" if latest['ma_20'] > latest['ma_50'] else "bearish"
                summary += f"A {cross_type} 20/50 MA crossover was recently observed. "
            
            if ma_50_200_cross:
                cross_type = "bullish" if latest['ma_50'] > latest['ma_200'] else "bearish"
                summary += f"A {cross_type} 50/200 MA crossover (golden/death cross) was recently observed. "
            
            return {
                "indicators": {
                    "ma_20": latest['ma_20'],
                    "ma_50": latest['ma_50'],
                    "ma_200": latest['ma_200'],
                    "macd": latest['macd'],
                    "macd_signal": latest['macd_signal'],
                    "macd_histogram": latest['macd_histogram'],
                    "rsi": latest['rsi'],
                    "bb_upper": latest['bb_upper'],
                    "bb_middle": latest['bb_middle'],
                    "bb_lower": latest['bb_lower']
                },
                "trend_analysis": {
                    "long_term": long_trend,
                    "medium_term": medium_trend,
                    "short_term": short_trend,
                    "trend_strength": trend_strength
                },
                "signals": indicator_signals,
                "support_resistance": support_resistance,
                "overall_bias": overall_bias,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "summary": summary
            }
        except Exception as e:
            self.logger.error(f"Error generating simple technical analysis: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _find_simple_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Find simple support and resistance levels.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            levels = []
            
            # Use the last 60 periods for analysis
            lookback = min(len(df), 60)
            
            # Use high and low values to identify levels
            highs = df['high'].iloc[-lookback:].values
            lows = df['low'].iloc[-lookback:].values
            close = df['close'].iloc[-1]
            
            # Function to find local maxima/minima
            def find_local_extrema(data, min_strength=2):
                extrema = []
                for i in range(1, len(data) - 1):
                    if data[i] > data[i-1] and data[i] > data[i+1]:
                        # Check strength
                        strength = 0
                        for j in range(max(0, i-5), min(len(data), i+6)):
                            if i != j and data[i] > data[j]:
                                strength += 1
                        
                        if strength >= min_strength:
                            extrema.append({"price": data[i], "strength": strength})
                
                return extrema
            
            # Find local maxima (potential resistance)
            resistance_levels = find_local_extrema(highs)
            
            # Find local minima (potential support)
            support_values = -lows
            support_candidates = find_local_extrema(support_values)
            support_levels = [{"price": -level["price"], "strength": level["strength"]} for level in support_candidates]
            
            # Tag levels as support or resistance
            for level in resistance_levels:
                level["type"] = "resistance"
                levels.append(level)
            
            for level in support_levels:
                level["type"] = "support"
                levels.append(level)
            
            # Sort by distance from current price
            for level in levels:
                level["distance"] = abs(level["price"] - close) / close * 100
            
            levels.sort(key=lambda x: x["distance"])
            
            # Keep top 3 levels of each type
            top_support = [level for level in levels if level["type"] == "support"][:3]
            top_resistance = [level for level in levels if level["type"] == "resistance"][:3]
            
            return {
                "support_levels": top_support,
                "resistance_levels": top_resistance
            }
            
        except Exception as e:
            self.logger.error(f"Error finding support/resistance levels: {e}")
            return {
                "support_levels": [],
                "resistance_levels": []
            }
    
    def _format_fundamental_analysis(self, fund_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format fundamental analysis data for the report.
        
        Args:
            fund_analysis: Fundamental analysis data
            
        Returns:
            Formatted fundamental analysis section
        """
        if not fund_analysis or "status" in fund_analysis and fund_analysis["status"] == "error":
            return {
                "status": "error",
                "message": fund_analysis.get("error", "Unknown error in fundamental analysis")
            }
        
        # Extract relevant sections
        formatted = {
            "financial_metrics": fund_analysis.get("financial_metrics", {}),
            "valuation_metrics": fund_analysis.get("valuation_metrics", {}),
            "growth_metrics": fund_analysis.get("growth_metrics", {}),
            "peer_comparison": fund_analysis.get("peer_comparison", {}),
            "summary": fund_analysis.get("summary", "")
        }
        
        return formatted
    
    def _get_financial_data(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get financial data from database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with financial data
        """
        try:
            # Query the financial data collection
            financial_data = self.db.financial_data_collection.find_one(
                {"symbol": symbol, "exchange": exchange},
                sort=[("timestamp", -1)]
            )
            
            if not financial_data:
                return {
                    "status": "not_available",
                    "message": "Financial data not available"
                }
            
            # Extract key metrics
            metrics = {}
            if "key_metrics" in financial_data:
                metrics = financial_data["key_metrics"]
            
            # Extract quarterly results
            quarterly_results = []
            if "quarterly_results" in financial_data:
                quarterly_results = financial_data["quarterly_results"]
            
            # Extract valuation metrics
            valuation = {}
            if "valuation_metrics" in financial_data:
                valuation = financial_data["valuation_metrics"]
            
            return {
                "key_metrics": metrics,
                "quarterly_results": quarterly_results,
                "valuation_metrics": valuation,
                "timestamp": financial_data.get("timestamp", datetime.now())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting financial data for {symbol}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_simple_volatility_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate simplified volatility analysis.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with volatility analysis
        """
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate volatility (standard deviation of returns)
            current_volatility = df['returns'].rolling(window=21).std().iloc[-1] * 100  # Convert to percentage
            
            # Historical volatility for comparison
            historical_volatility = df['returns'].rolling(window=21).std().iloc[-63:-1].mean() * 100
            
            # Compare current to historical
            volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1
            
            # Determine volatility state
            if volatility_ratio > 1.5:
                volatility_state = "expanding"
            elif volatility_ratio < 0.7:
                volatility_state = "contracting"
            else:
                volatility_state = "stable"
            
            # Calculate ATR
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(window=14).mean()
            
            latest_atr = df['atr'].iloc[-1]
            atr_pct = latest_atr / df['close'].iloc[-1] * 100
            
            # Generate summary text
            summary = f"Current 21-day volatility is {current_volatility:.2f}%, which is "
            
            if volatility_ratio > 1.2:
                summary += f"higher than the historical average. "
            elif volatility_ratio < 0.8:
                summary += f"lower than the historical average. "
            else:
                summary += f"in line with the historical average. "
            
            summary += f"Volatility is currently {volatility_state}. "
            summary += f"The Average True Range (ATR) is {latest_atr:.2f} points or {atr_pct:.2f}% of price."
            
            return {
                "current_volatility": current_volatility,
                "historical_volatility": historical_volatility,
                "volatility_ratio": volatility_ratio,
                "volatility_state": volatility_state,
                "atr": latest_atr,
                "atr_percent": atr_pct,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating simple volatility analysis: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_price_levels_section(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate price levels section.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with price levels
        """
        try:
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Find support and resistance levels
            sr_levels = self._find_simple_support_resistance(df)
            
            # Add pivot points
            pivot_points = self._calculate_pivot_points(df)
            
            # Add Fibonacci retracement levels
            fib_levels = self._calculate_fibonacci_levels(df)
            
            # Add moving averages as potential support/resistance
            ma_levels = self._get_moving_average_levels(df)
            
            # Generate summary text
            summary = self._generate_price_levels_summary(
                current_price, sr_levels, pivot_points, fib_levels, ma_levels
            )
            
            return {
                "current_price": current_price,
                "support_resistance": sr_levels,
                "pivot_points": pivot_points,
                "fibonacci_levels": fib_levels,
                "moving_averages": ma_levels,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating price levels section: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate pivot points based on the previous day's data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with pivot points
        """
        try:
            # Get previous day's data
            prev_high = df['high'].iloc[-2]
            prev_low = df['low'].iloc[-2]
            prev_close = df['close'].iloc[-2]
            
            # Calculate pivot point
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # Calculate support levels
            s1 = 2 * pivot - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = s2 - (prev_high - prev_low)
            
            # Calculate resistance levels
            r1 = 2 * pivot - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = r2 + (prev_high - prev_low)
            
            return {
                "pivot": pivot,
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "r1": r1,
                "r2": r2,
                "r3": r3
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {e}")
            return {}
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with Fibonacci levels
        """
        try:
            # Find recent high and low (last 20 periods)
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            
            # If price is closer to recent high, calculate retracements from high to low
            current_price = df['close'].iloc[-1]
            
            if abs(current_price - recent_high) < abs(current_price - recent_low):
                # Downtrend case - retracements from high to low
                diff = recent_high - recent_low
                
                return {
                    "trend": "downtrend",
                    "0.0": recent_low,
                    "0.236": recent_low + 0.236 * diff,
                    "0.382": recent_low + 0.382 * diff,
                    "0.5": recent_low + 0.5 * diff,
                    "0.618": recent_low + 0.618 * diff,
                    "0.786": recent_low + 0.786 * diff,
                    "1.0": recent_high
                }
            else:
                # Uptrend case - retracements from low to high
                diff = recent_high - recent_low
                
                return {
                    "trend": "uptrend",
                    "0.0": recent_low,
                    "0.236": recent_low + 0.236 * diff,
                    "0.382": recent_low + 0.382 * diff,
                    "0.5": recent_low + 0.5 * diff,
                    "0.618": recent_low + 0.618 * diff,
                    "0.786": recent_low + 0.786 * diff,
                    "1.0": recent_high
                }
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}
    
    def _get_moving_average_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get moving average levels as potential support/resistance.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with moving average levels
        """
        ma_levels = {}
        
        try:
            # Common moving averages
            periods = [20, 50, 100, 200]
            
            for period in periods:
                if len(df) >= period:
                    ma = df['close'].rolling(window=period).mean().iloc[-1]
                    ma_levels[f"MA{period}"] = ma
            
            return ma_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating moving average levels: {e}")
            return {}
    
    def _generate_price_levels_summary(self, current_price: float, 
                                    sr_levels: Dict[str, Any],
                                    pivot_points: Dict[str, float],
                                    fib_levels: Dict[str, float],
                                    ma_levels: Dict[str, float]) -> str:
        """
        Generate summary text for price levels section.
        
        Args:
            current_price: Current price
            sr_levels: Support and resistance levels
            pivot_points: Pivot points
            fib_levels: Fibonacci retracement levels
            ma_levels: Moving average levels
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Find nearest support and resistance
        nearest_support = None
        support_distance = float('inf')
        
        for level in sr_levels.get("support_levels", []):
            price = level.get("price", 0)
            if price < current_price and current_price - price < support_distance:
                support_distance = current_price - price
                nearest_support = price
        
        nearest_resistance = None
        resistance_distance = float('inf')
        
        for level in sr_levels.get("resistance_levels", []):
            price = level.get("price", 0)
            if price > current_price and price - current_price < resistance_distance:
                resistance_distance = price - current_price
                nearest_resistance = price
        
        # Mention nearest levels
        if nearest_support:
            support_pct = (nearest_support / current_price - 1) * 100
            summary_parts.append(f"Nearest support is at {nearest_support:.2f} ({abs(support_pct):.2f}% below current price).")
        
        if nearest_resistance:
            resistance_pct = (nearest_resistance / current_price - 1) * 100
            summary_parts.append(f"Nearest resistance is at {nearest_resistance:.2f} ({resistance_pct:.2f}% above current price).")
        
        # Mention pivot points
        if "pivot" in pivot_points:
            pivot = pivot_points["pivot"]
            pivot_relation = "above" if current_price > pivot else "below"
            summary_parts.append(f"Price is {pivot_relation} the pivot point ({pivot:.2f}).")
            
            # Check if price is between specific pivot levels
            if "s1" in pivot_points and "r1" in pivot_points:
                s1 = pivot_points["s1"]
                r1 = pivot_points["r1"]
                
                if current_price > s1 and current_price < r1:
                    summary_parts.append(f"Price is between S1 ({s1:.2f}) and R1 ({r1:.2f}).")
        
        # Mention Fibonacci levels
        if fib_levels and "trend" in fib_levels:
            trend = fib_levels["trend"]
            
            # Find the Fibonacci levels that price is between
            fib_levels_list = [(float(k), v) for k, v in fib_levels.items() if k != "trend"]
            fib_levels_list.sort()
            
            for i in range(len(fib_levels_list) - 1):
                lower_level = fib_levels_list[i]
                upper_level = fib_levels_list[i + 1]
                
                if lower_level[1] <= current_price <= upper_level[1]:
                    summary_parts.append(f"Price is between the {lower_level[0]} ({lower_level[1]:.2f}) and {upper_level[0]} ({upper_level[1]:.2f}) Fibonacci retracement levels.")
                    break
        
        # Mention moving averages
        ma_above = []
        ma_below = []
        
        for ma_name, ma_value in ma_levels.items():
            if ma_value > current_price:
                ma_above.append((ma_name, ma_value))
            else:
                ma_below.append((ma_name, ma_value))
        
        if ma_above:
            ma_above.sort(key=lambda x: x[1])
            nearest_ma_above = ma_above[0]
            summary_parts.append(f"The {nearest_ma_above[0]} at {nearest_ma_above[1]:.2f} is acting as resistance.")
        
        if ma_below:
            ma_below.sort(key=lambda x: x[1], reverse=True)
            nearest_ma_below = ma_below[0]
            summary_parts.append(f"The {nearest_ma_below[0]} at {nearest_ma_below[1]:.2f} is acting as support.")
        
        # Add risk-reward assessment
        if nearest_support and nearest_resistance:
            potential_loss = current_price - nearest_support
            potential_gain = nearest_resistance - current_price
            risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
            
            if risk_reward >= 2:
                summary_parts.append(f"Risk-reward ratio of {risk_reward:.2f} is favorable for long positions.")
            elif risk_reward <= 0.5:
                summary_parts.append(f"Risk-reward ratio of {risk_reward:.2f} is favorable for short positions.")
            else:
                summary_parts.append(f"Risk-reward ratio of {risk_reward:.2f} is neutral.")
        
        return " ".join(summary_parts)
    
    def _generate_simple_opportunities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate simplified trading opportunities.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with trading opportunities
        """
        try:
            # Calculate basic indicators if not already present
            if 'ma_20' not in df.columns:
                df['ma_20'] = df['close'].rolling(window=20).mean()
            
            if 'ma_50' not in df.columns:
                df['ma_50'] = df['close'].rolling(window=50).mean()
            
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            if 'bb_upper' not in df.columns:
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                df['bb_std'] = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
                df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            
            # Look for common patterns/setups
            opportunities = []
            
            # 1. Moving Average Crossover
            if len(df) > 2:
                if df['ma_20'].iloc[-2] <= df['ma_50'].iloc[-2] and df['ma_20'].iloc[-1] > df['ma_50'].iloc[-1]:
                    opportunities.append({
                        "type": "Moving Average Crossover",
                        "direction": "bullish",
                        "description": "The 20-day moving average has crossed above the 50-day moving average, indicating potential upward momentum.",
                        "score": 3
                    })
                elif df['ma_20'].iloc[-2] >= df['ma_50'].iloc[-2] and df['ma_20'].iloc[-1] < df['ma_50'].iloc[-1]:
                    opportunities.append({
                        "type": "Moving Average Crossover",
                        "direction": "bearish",
                        "description": "The 20-day moving average has crossed below the 50-day moving average, indicating potential downward momentum.",
                        "score": 3
                    })
            
            # 2. RSI Oversold/Overbought
            if df['rsi'].iloc[-1] < 30:
                opportunities.append({
                    "type": "Oversold Condition",
                    "direction": "bullish",
                    "description": f"RSI is in oversold territory at {df['rsi'].iloc[-1]:.2f}, suggesting a potential bullish reversal.",
                    "score": 3
                })
            elif df['rsi'].iloc[-1] > 70:
                opportunities.append({
                    "type": "Overbought Condition",
                    "direction": "bearish",
                    "description": f"RSI is in overbought territory at {df['rsi'].iloc[-1]:.2f}, suggesting a potential bearish reversal.",
                    "score": 3
                })
            
            # 3. Bollinger Band Touch
            if df['close'].iloc[-1] < df['bb_lower'].iloc[-1]:
                opportunities.append({
                    "type": "Bollinger Band Touch",
                    "direction": "bullish",
                    "description": "Price has touched or broken below the lower Bollinger Band, suggesting a potential bullish reversal.",
                    "score": 2
                })
            elif df['close'].iloc[-1] > df['bb_upper'].iloc[-1]:
                opportunities.append({
                    "type": "Bollinger Band Touch",
                    "direction": "bearish",
                    "description": "Price has touched or broken above the upper Bollinger Band, suggesting a potential bearish reversal.",
                    "score": 2
                })
            
            # 4. Support/Resistance Test
            sr_levels = self._find_simple_support_resistance(df)
            
            current_price = df['close'].iloc[-1]
            
            for level in sr_levels.get("support_levels", []):
                price = level.get("price", 0)
                if abs(current_price - price) / current_price < 0.02:  # Within 2%
                    opportunities.append({
                        "type": "Support Test",
                        "direction": "bullish",
                        "description": f"Price is testing a support level at {price:.2f}.",
                        "score": 2
                    })
                    break  # Just use the nearest one
            
            for level in sr_levels.get("resistance_levels", []):
                price = level.get("price", 0)
                if abs(current_price - price) / current_price < 0.02:  # Within 2%
                    opportunities.append({
                        "type": "Resistance Test",
                        "direction": "bearish",
                        "description": f"Price is testing a resistance level at {price:.2f}.",
                        "score": 2
                    })
                    break  # Just use the nearest one
            
            # Sort by score (highest first)
            opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Generate summary text
            summary = ""
            if opportunities:
                summary = f"Identified {len(opportunities)} potential trading opportunities. "
                summary += f"The highest-scoring opportunity is a {opportunities[0]['type']} setup with a {opportunities[0]['direction']} bias. "
                summary += opportunities[0]['description']
            else:
                summary = "No clear trading opportunities identified at the current price level."
            
            return {
                "opportunities": opportunities,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating simple opportunities: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _format_correlation_analysis(self, corr_analysis: Dict[str, Any], 
                                  focus_symbol: str) -> Dict[str, Any]:
        """
        Format correlation analysis data for the report.
        
        Args:
            corr_analysis: Correlation analysis data
            focus_symbol: Symbol to focus on
            
        Returns:
            Formatted correlation analysis section
        """
        if not corr_analysis or "status" in corr_analysis and corr_analysis["status"] == "error":
            return {
                "status": "error",
                "message": corr_analysis.get("error", "Unknown error in correlation analysis")
            }
        
        # Extract correlations specific to the focus symbol
        symbol_correlations = []
        
        correlation_matrix = corr_analysis.get("correlation_matrix", [])
        for row in correlation_matrix:
            if row.get("symbol") == focus_symbol:
                # Extract correlations with other symbols
                symbol_data = {k: v for k, v in row.items() if k != "symbol"}
                
                # Convert to list format
                for other_symbol, correlation in symbol_data.items():
                    symbol_correlations.append({
                        "symbol": other_symbol,
                        "correlation": correlation
                    })
                
                # Sort by absolute correlation (highest first)
                symbol_correlations.sort(key=lambda x: abs(x.get("correlation", 0)), reverse=True)
                break
        
        # Extract relevant parts of the correlation analysis
        formatted = {
            "symbol_correlations": symbol_correlations,
            "high_correlation_pairs": [
                pair for pair in corr_analysis.get("high_correlation_pairs", [])
                if pair.get("symbol1") == focus_symbol or pair.get("symbol2") == focus_symbol
            ],
            "inverse_correlation_pairs": [
                pair for pair in corr_analysis.get("inverse_correlation_pairs", [])
                if pair.get("symbol1") == focus_symbol or pair.get("symbol2") == focus_symbol
            ],
            "correlation_stability": corr_analysis.get("correlation_stability", {}),
            "summary": corr_analysis.get("correlation_summary", "")
        }
        
        # Generate focused summary if original doesn't focus on this symbol
        if focus_symbol not in formatted["summary"]:
            formatted["summary"] = self._generate_focused_correlation_summary(
                focus_symbol, symbol_correlations,
                formatted["high_correlation_pairs"],
                formatted["inverse_correlation_pairs"]
            )
        
        return formatted
    
    def _generate_focused_correlation_summary(self, symbol: str, 
                                          symbol_correlations: List[Dict[str, Any]],
                                          high_pairs: List[Dict[str, Any]],
                                          inverse_pairs: List[Dict[str, Any]]) -> str:
        """
        Generate correlation summary focused on a specific symbol.
        
        Args:
            symbol: Focus symbol
            symbol_correlations: Correlations with other symbols
            high_pairs: Highly correlated pairs
            inverse_pairs: Inversely correlated pairs
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Overall correlations
        if symbol_correlations:
            # Calculate average correlation
            avg_corr = sum(abs(item.get("correlation", 0)) for item in symbol_correlations) / len(symbol_correlations)
            
            if avg_corr > 0.7:
                summary_parts.append(f"{symbol} shows strong overall correlation with the analyzed securities.")
            elif avg_corr > 0.4:
                summary_parts.append(f"{symbol} shows moderate overall correlation with the analyzed securities.")
            else:
                summary_parts.append(f"{symbol} shows low overall correlation with the analyzed securities, indicating good diversification potential.")
        
        # Strongest direct correlation
        if high_pairs:
            pair = high_pairs[0]
            other_symbol = pair.get("symbol1") if pair.get("symbol2") == symbol else pair.get("symbol2")
            correlation = pair.get("correlation", 0)
            
            summary_parts.append(f"{symbol} has the strongest positive correlation with {other_symbol} ({correlation:.2f}).")
        
        # Strongest inverse correlation
        if inverse_pairs:
            pair = inverse_pairs[0]
            other_symbol = pair.get("symbol1") if pair.get("symbol2") == symbol else pair.get("symbol2")
            correlation = pair.get("correlation", 0)
            
            summary_parts.append(f"{symbol} has the strongest inverse correlation with {other_symbol} ({correlation:.2f}), which may provide hedging potential.")
        
        # Trading implications
        if high_pairs or inverse_pairs:
            summary_parts.append("These correlation relationships can be utilized for pair trading strategies or portfolio diversification.")
        
        return " ".join(summary_parts)
    
    def _generate_summary_section(self, symbol: str, exchange: str, 
                               sections: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary and recommendation section for stock report.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            sections: Report sections
            
        Returns:
            Dictionary with summary and recommendation
        """
        try:
            # Extract key insights from each section
            overview = sections.get("overview", {})
            technical = sections.get("technical_analysis", {})
            fundamental = sections.get("fundamental_analysis", {})
            volatility = sections.get("volatility_analysis", {})
            opportunities = sections.get("opportunities", {})
            
            # Determine overall sentiment
            sentiments = []
            
            # Technical sentiment
            if "overall_bias" in technical:
                technical_bias = technical["overall_bias"]
                sentiments.append(technical_bias)
            
            # Fundamental sentiment
            fundamental_sentiment = "neutral"
            if isinstance(fundamental, dict):
                # Try to extract sentiment from valuation metrics or key metrics
                pe_ratio = None
                industry_pe = None
                
                if "valuation_metrics" in fundamental:
                    valuation = fundamental["valuation_metrics"]
                    pe_ratio = valuation.get("pe_ratio")
                    industry_pe = valuation.get("industry_pe")
                
                if pe_ratio is not None and industry_pe is not None:
                    if pe_ratio < industry_pe * 0.8:
                        fundamental_sentiment = "bullish"  # Undervalued
                    elif pe_ratio > industry_pe * 1.2:
                        fundamental_sentiment = "bearish"  # Overvalued
            
            sentiments.append(fundamental_sentiment)
            
            # Opportunity sentiment
            opportunity_sentiment = "neutral"
            if isinstance(opportunities, dict):
                opportunity_list = opportunities.get("opportunities", [])
                
                if opportunity_list:
                    bullish_count = sum(1 for opp in opportunity_list if opp.get("direction") == "bullish")
                    bearish_count = sum(1 for opp in opportunity_list if opp.get("direction") == "bearish")
                    
                    if bullish_count > bearish_count:
                        opportunity_sentiment = "bullish"
                    elif bearish_count > bullish_count:
                        opportunity_sentiment = "bearish"
            
            sentiments.append(opportunity_sentiment)
            
            # Price level sentiment
            price_levels = sections.get("price_levels", {})
            price_level_sentiment = "neutral"
            
            if isinstance(price_levels, dict):
                current_price = price_levels.get("current_price", 0)
                sr_levels = price_levels.get("support_resistance", {})
                
                support_levels = sr_levels.get("support_levels", [])
                resistance_levels = sr_levels.get("resistance_levels", [])
                
                if support_levels and resistance_levels:
                    nearest_support = min([abs(current_price - level.get("price", 0)) for level in support_levels])
                    nearest_resistance = min([abs(current_price - level.get("price", 0)) for level in resistance_levels])
                    
                    if nearest_support < nearest_resistance:
                        price_level_sentiment = "bullish"  # Closer to support
                    elif nearest_resistance < nearest_support:
                        price_level_sentiment = "bearish"  # Closer to resistance
            
            sentiments.append(price_level_sentiment)
            
            # Count sentiments
            bullish_count = sentiments.count("bullish")
            bearish_count = sentiments.count("bearish")
            neutral_count = sentiments.count("neutral")
            
            # Determine overall sentiment
            if bullish_count > bearish_count:
                overall_sentiment = "bullish"
            elif bearish_count > bullish_count:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"
            
            # Generate timeframe recommendation
            timeframe_recommendation = self._generate_timeframe_recommendation(
                overall_sentiment, volatility, technical, opportunities
            )
            
            # Generate risk assessment
            risk_assessment = self._generate_risk_assessment(
                volatility, price_levels, technical
            )
            
            # Generate strategy recommendation
            strategy_recommendation = self._generate_strategy_recommendation(
                overall_sentiment, volatility, technical, opportunities
            )
            
            # Generate summary text
            summary_text = self._generate_summary_text(
                symbol, overall_sentiment, technical, fundamental, 
                volatility, opportunities, timeframe_recommendation, 
                risk_assessment, strategy_recommendation
            )
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_breakdown": {
                    "technical": technical.get("overall_bias", "neutral"),
                    "fundamental": fundamental_sentiment,
                    "opportunities": opportunity_sentiment,
                    "price_levels": price_level_sentiment
                },
                "timeframe_recommendation": timeframe_recommendation,
                "risk_assessment": risk_assessment,
                "strategy_recommendation": strategy_recommendation,
                "summary": summary_text
            }
            
        except Exception as e:
            self.logger.error(f"Error generating summary section: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_timeframe_recommendation(self, overall_sentiment: str,
                                        volatility: Dict[str, Any],
                                        technical: Dict[str, Any],
                                        opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate timeframe recommendation.
        
        Args:
            overall_sentiment: Overall sentiment
            volatility: Volatility analysis
            technical: Technical analysis
            opportunities: Trading opportunities
            
        Returns:
            Dictionary with timeframe recommendation
        """
        # Default recommendation
        timeframe = "medium-term"
        reasoning = "Based on balanced risk-reward and market conditions."
        
        # Adjust based on volatility
        volatility_state = volatility.get("volatility_state", "stable")
        volatility_regime = volatility.get("volatility_regime", "normal")
        
        if volatility_state == "expanding" or volatility_regime == "high":
            timeframe = "short-term"
            reasoning = "High or expanding volatility suggests shorter holding periods to manage risk."
        elif volatility_state == "contracting" and volatility_regime == "low":
            timeframe = "long-term"
            reasoning = "Low and contracting volatility suggests favorable conditions for longer-term positions."
        
        # Adjust based on technical trend
        if "trend_analysis" in technical:
            trend = technical["trend_analysis"]
            
            long_term = trend.get("long_term", "neutral")
            medium_term = trend.get("medium_term", "neutral")
            short_term = trend.get("short_term", "neutral")
            
            if long_term == medium_term == short_term and long_term != "neutral":
                timeframe = "multiple timeframes"
                reasoning = f"Aligned {long_term} trends across all timeframes suggest potential for both short and long-term strategies."
            elif short_term != medium_term:
                timeframe = "short-term"
                reasoning = "Divergence between short and medium-term trends suggests focusing on shorter timeframes."
        
        # Adjust based on opportunities
        if isinstance(opportunities, dict):
            opportunity_list = opportunities.get("opportunities", [])
            
            if opportunity_list:
                # Check for strong short-term signals
                short_term_signals = ["Oversold Condition", "Overbought Condition", 
                                      "Bollinger Band Touch", "Support Test", "Resistance Test"]
                
                short_term_count = sum(1 for opp in opportunity_list 
                                     if opp.get("type") in short_term_signals)
                
                if short_term_count > 0 and (volatility_state == "expanding" or volatility_regime == "high"):
                    timeframe = "short-term"
                    reasoning = "Strong short-term signals combined with current volatility conditions favor shorter timeframes."
        
        return {
            "timeframe": timeframe,
            "reasoning": reasoning
        }
    
    def _generate_risk_assessment(self, volatility: Dict[str, Any],
                               price_levels: Dict[str, Any],
                               technical: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate risk assessment.
        
        Args:
            volatility: Volatility analysis
            price_levels: Price levels analysis
            technical: Technical analysis
            
        Returns:
            Dictionary with risk assessment
        """
        # Default assessment
        risk_level = "moderate"
        key_risks = []
        risk_mitigation = []
        
        # Assess based on volatility
        if isinstance(volatility, dict):
            volatility_state = volatility.get("volatility_state", "stable")
            volatility_regime = volatility.get("volatility_regime", "normal")
            
            if volatility_state == "expanding" and volatility_regime == "high":
                risk_level = "high"
                key_risks.append("High and expanding volatility")
                risk_mitigation.append("Use tighter stops and reduced position sizes")
            elif volatility_state == "contracting" and volatility_regime == "low":
                risk_level = "low"
                key_risks.append("Potential for volatility expansion from low levels")
                risk_mitigation.append("Be alert for volatility breakouts")
            
            # Add ATR-based risk
            atr_percent = volatility.get("atr_percent")
            if atr_percent is not None:
                if atr_percent > 3:
                    risk_level = "high"
                    key_risks.append(f"Daily range averaging {atr_percent:.2f}% of price")
                    risk_mitigation.append("Use wider stops to accommodate daily volatility")
        
        # Assess based on price levels
        if isinstance(price_levels, dict):
            sr_levels = price_levels.get("support_resistance", {})
            
            support_levels = sr_levels.get("support_levels", [])
            resistance_levels = sr_levels.get("resistance_levels", [])
            
            if not support_levels:
                key_risks.append("No clear support levels identified")
                risk_mitigation.append("Use technical indicators for stop placement")
            
            if not resistance_levels:
                key_risks.append("No clear resistance levels identified")
                risk_mitigation.append("Use trailing stops or take profits based on volatility")
        
        # Assess based on technical conditions
        if isinstance(technical, dict):
            # Check for trend strength
            if "trend_analysis" in technical:
                trend_strength = technical["trend_analysis"].get("trend_strength", "moderate")
                
                if trend_strength == "weak":
                    key_risks.append("Weak or unclear trend direction")
                    risk_mitigation.append("Use smaller position sizes until trend strengthens")
            
            # Check for mixed signals
            bullish_count = technical.get("bullish_count", 0)
            bearish_count = technical.get("bearish_count", 0)
            
            if bullish_count > 0 and bearish_count > 0 and abs(bullish_count - bearish_count) <= 2:
                key_risks.append("Mixed technical signals")
                risk_mitigation.append("Wait for clearer signal alignment before entry")
        
        # If no specific risks identified, add general market risk
        if not key_risks:
            key_risks.append("General market risk")
            risk_mitigation.append("Follow proper position sizing and risk management")
        
        return {
            "risk_level": risk_level,
            "key_risks": key_risks,
            "risk_mitigation": risk_mitigation
        }
    
    def _generate_strategy_recommendation(self, overall_sentiment: str,
                                       volatility: Dict[str, Any],
                                       technical: Dict[str, Any],
                                       opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate strategy recommendation.
        
        Args:
            overall_sentiment: Overall sentiment
            volatility: Volatility analysis
            technical: Technical analysis
            opportunities: Trading opportunities
            
        Returns:
            Dictionary with strategy recommendation
        """
        # Default recommendation
        strategy_type = "wait for clearer signals"
        entry_strategy = "No entry recommended at this time"
        exit_strategy = "N/A"
        
        # Recommendations for different sentiments
        if overall_sentiment == "bullish":
            if isinstance(opportunities, dict):
                opportunity_list = opportunities.get("opportunities", [])
                bullish_opportunities = [opp for opp in opportunity_list if opp.get("direction") == "bullish"]
                
                if bullish_opportunities:
                    # Use the highest-scored opportunity
                    top_opportunity = max(bullish_opportunities, key=lambda x: x.get("score", 0))
                    opp_type = top_opportunity.get("type", "")
                    
                    if "Moving Average Crossover" in opp_type:
                        strategy_type = "trend following"
                        entry_strategy = "Enter on pullback to the faster moving average"
                        exit_strategy = "Exit on moving average crossover reversal or trailing stop"
                    elif "Oversold Condition" in opp_type or "Bollinger Band Touch" in opp_type:
                        strategy_type = "mean reversion"
                        entry_strategy = "Enter on first sign of reversal from oversold levels"
                        exit_strategy = "Exit at the mean or when momentum stalls"
                    elif "Support Test" in opp_type:
                        strategy_type = "support bounce"
                        entry_strategy = "Enter on confirmation of support holding"
                        exit_strategy = "Exit at the next resistance level or if support breaks"
                    else:
                        strategy_type = "trend following"
                        entry_strategy = "Enter on bullish confirmation signals"
                        exit_strategy = "Use trailing stops based on volatility"
                else:
                    strategy_type = "trend following"
                    entry_strategy = "Enter on pullbacks in the bullish trend"
                    exit_strategy = "Use trailing stops to protect profits"
            else:
                strategy_type = "trend following"
                entry_strategy = "Enter on pullbacks in the bullish trend"
                exit_strategy = "Use trailing stops to protect profits"
        
        elif overall_sentiment == "bearish":
            if isinstance(opportunities, dict):
                opportunity_list = opportunities.get("opportunities", [])
                bearish_opportunities = [opp for opp in opportunity_list if opp.get("direction") == "bearish"]
                
                if bearish_opportunities:
                    # Use the highest-scored opportunity
                    top_opportunity = max(bearish_opportunities, key=lambda x: x.get("score", 0))
                    opp_type = top_opportunity.get("type", "")
                    
                    if "Moving Average Crossover" in opp_type:
                        strategy_type = "trend following"
                        entry_strategy = "Enter on rallies to the faster moving average"
                        exit_strategy = "Exit on moving average crossover reversal or trailing stop"
                    elif "Overbought Condition" in opp_type or "Bollinger Band Touch" in opp_type:
                        strategy_type = "mean reversion"
                        entry_strategy = "Enter on first sign of reversal from overbought levels"
                        exit_strategy = "Exit at the mean or when momentum stalls"
                    elif "Resistance Test" in opp_type:
                        strategy_type = "resistance rejection"
                        entry_strategy = "Enter on confirmation of resistance rejection"
                        exit_strategy = "Exit at the next support level or if resistance breaks"
                    else:
                        strategy_type = "trend following"
                        entry_strategy = "Enter on bearish confirmation signals"
                        exit_strategy = "Use trailing stops based on volatility"
                else:
                    strategy_type = "trend following"
                    entry_strategy = "Enter on rallies in the bearish trend"
                    exit_strategy = "Use trailing stops to protect profits"
            else:
                strategy_type = "trend following"
                entry_strategy = "Enter on rallies in the bearish trend"
                exit_strategy = "Use trailing stops to protect profits"
        
        # Adjust strategy based on volatility
        if isinstance(volatility, dict):
            volatility_state = volatility.get("volatility_state", "stable")
            volatility_regime = volatility.get("volatility_regime", "normal")
            
            if volatility_state == "expanding" or volatility_regime == "high":
                # Add volatility considerations to strategy
                if "trend following" in strategy_type:
                    exit_strategy += " and consider tighter stops due to high volatility"
                elif "mean reversion" in strategy_type:
                    entry_strategy += " and use smaller position sizes due to high volatility"
            
            if volatility_state == "contracting" and volatility_regime == "low":
                if "trend following" in strategy_type:
                    entry_strategy += " and consider breakout strategies as volatility expands"
                elif "mean reversion" in strategy_type:
                    entry_strategy += " but be aware that low volatility may limit mean reversion potential"
        
        return {
            "strategy_type": strategy_type,
            "entry_strategy": entry_strategy,
            "exit_strategy": exit_strategy
        }
    
    def _generate_summary_text(self, symbol: str, overall_sentiment: str,
                            technical: Dict[str, Any], fundamental: Dict[str, Any],
                            volatility: Dict[str, Any], opportunities: Dict[str, Any],
                            timeframe: Dict[str, Any], risk: Dict[str, Any], 
                            strategy: Dict[str, Any]) -> str:
        """
        Generate summary text for the report.
        
        Args:
            symbol: Stock symbol
            overall_sentiment: Overall sentiment
            technical: Technical analysis
            fundamental: Fundamental analysis
            volatility: Volatility analysis
            opportunities: Trading opportunities
            timeframe: Timeframe recommendation
            risk: Risk assessment
            strategy: Strategy recommendation
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Overall assessment
        sentiment_str = overall_sentiment.capitalize()
        summary_parts.append(f"Overall outlook for {symbol} is {sentiment_str}.")
        
        # Technical summary
        if isinstance(technical, dict) and "summary" in technical:
            tech_summary = technical["summary"]
            if tech_summary:
                # Extract first sentence of technical summary
                first_sentence = tech_summary.split('.')[0] + '.'
                summary_parts.append(f"Technically, {first_sentence.lower()}")
        
        # Fundamental summary
        if isinstance(fundamental, dict) and "summary" in fundamental:
            fund_summary = fundamental["summary"]
            if fund_summary:
                # Extract first sentence of fundamental summary
                first_sentence = fund_summary.split('.')[0] + '.'
                summary_parts.append(f"Fundamentally, {first_sentence.lower()}")
        
        # Volatility summary
        if isinstance(volatility, dict) and "summary" in volatility:
            vol_summary = volatility["summary"]
            if vol_summary:
                # Extract first sentence of volatility summary
                first_sentence = vol_summary.split('.')[0] + '.'
                summary_parts.append(first_sentence)
        
        # Opportunities summary
        if isinstance(opportunities, dict) and "summary" in opportunities:
            opp_summary = opportunities["summary"]
            if opp_summary:
                summary_parts.append(opp_summary)
        
        # Timeframe recommendation
        if timeframe:
            timeframe_str = timeframe.get("timeframe", "medium-term")
            reasoning = timeframe.get("reasoning", "")
            summary_parts.append(f"The recommended trading timeframe is {timeframe_str}. {reasoning}")
        
        # Risk assessment
        if risk:
            risk_level = risk.get("risk_level", "moderate")
            key_risks = risk.get("key_risks", [])
            risk_mitigation = risk.get("risk_mitigation", [])
            
            risk_str = f"Risk assessment: {risk_level.capitalize()}."
            if key_risks:
                risk_str += f" Key risks include {', '.join(key_risks).lower()}."
            if risk_mitigation:
                risk_str += f" Consider {', '.join(risk_mitigation).lower()}."
            
            summary_parts.append(risk_str)
        
        # Strategy recommendation
        if strategy:
            strategy_type = strategy.get("strategy_type", "")
            entry_strategy = strategy.get("entry_strategy", "")
            exit_strategy = strategy.get("exit_strategy", "")
            
            if strategy_type != "wait for clearer signals":
                summary_parts.append(f"Recommended strategy: {strategy_type.capitalize()}. {entry_strategy}. {exit_strategy}.")
            else:
                summary_parts.append(f"Recommendation: {entry_strategy}.")
        
        return " ".join(summary_parts)
    
    def _generate_stock_charts(self, df: pd.DataFrame, symbol: str) -> Dict[str, str]:
        """
        Generate charts for stock report.
        
        Args:
            df: DataFrame with market data
            symbol: Stock symbol
            
        Returns:
            Dictionary with chart data
        """
        charts = {}
        
        try:
            # 1. Price Chart with Moving Averages
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot price
            ax.plot(df['timestamp'], df['close'], label='Close Price')
            
            # Add moving averages
            if len(df) >= 20:
                ma20 = df['close'].rolling(window=20).mean()
                ax.plot(df['timestamp'], ma20, label='20-day MA')
            
            if len(df) >= 50:
                ma50 = df['close'].rolling(window=50).mean()
                ax.plot(df['timestamp'], ma50, label='50-day MA')
            
            if len(df) >= 200:
                ma200 = df['close'].rolling(window=200).mean()
                ax.plot(df['timestamp'], ma200, label='200-day MA')
            
            ax.set_title(f'{symbol} Price with Moving Averages')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            charts['price_chart'] = img_str
            plt.close(fig)
            
            # 2. Technical Indicators Chart (RSI, MACD)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
            
            # Calculate RSI if not already in DataFrame
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            # Plot RSI
            ax1.plot(df['timestamp'], df['rsi'], color='purple')
            ax1.axhline(y=70, color='r', linestyle='--')
            ax1.axhline(y=30, color='g', linestyle='--')
            ax1.set_title(f'{symbol} RSI (14)')
            ax1.set_ylabel('RSI')
            ax1.set_ylim(0, 100)
            ax1.grid(True)
            
            # Calculate MACD if not already in DataFrame
            if 'macd' not in df.columns:
                df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
                df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = df['ema_12'] - df['ema_26']
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Plot MACD
            ax2.plot(df['timestamp'], df['macd'], label='MACD')
            ax2.plot(df['timestamp'], df['macd_signal'], label='Signal')
            ax2.bar(df['timestamp'], df['macd_histogram'], label='Histogram', alpha=0.5)
            ax2.set_title(f'{symbol} MACD')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('MACD')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            charts['indicators_chart'] = img_str
            plt.close(fig)
            
            # 3. Volatility Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate volatility if not already in DataFrame
            if 'volatility' not in df.columns:
                df['volatility'] = df['close'].pct_change().rolling(window=21).std() * 100
            
            # Plot volatility
            ax.plot(df['timestamp'], df['volatility'], color='red')
            ax.set_title(f'{symbol} 21-day Volatility')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility (%)')
            ax.grid(True)
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            charts['volatility_chart'] = img_str
            plt.close(fig)
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
            return {}
    
    def _format_stock_report(self, report: Dict[str, Any]) -> str:
        """
        Format stock report as HTML.
        
        Args:
            report: Report data
            
        Returns:
            HTML formatted report
        """
        try:
            # Get report sections
            sections = report.get("sections", {})
            symbol = report.get("symbol", "")
            exchange = report.get("exchange", "")
            generated_at = report.get("generated_at", datetime.now())
            
            # Start building HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Stock Analysis Report: {symbol}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                    h1, h2, h3, h4 {{ color: #2c3e50; }}
                    .report-header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                    .chart {{ margin: 20px 0; text-align: center; }}
                    .chart img {{ max-width: 100%; height: auto; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .bullish {{ color: green; }}
                    .bearish {{ color: red; }}
                    .neutral {{ color: #888; }}
                    .summary-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1>Stock Analysis Report: {symbol} ({exchange})</h1>
                    <p>Generated on {generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """
            
            # 1. Overview Section
            if "overview" in sections:
                overview = sections["overview"]
                current_price = overview.get("current_price", 0)
                price_changes = overview.get("price_changes", {})
                day_change = price_changes.get("daily", 0)
                
                day_change_class = "neutral"
                if day_change > 0:
                    day_change_class = "bullish"
                elif day_change < 0:
                    day_change_class = "bearish"
                
                html += f"""
                <div class="section">
                    <h2>Overview</h2>
                    <p>{overview.get("summary", "")}</p>
                    
                    <table>
                        <tr>
                            <th>Current Price</th>
                            <td>₹{current_price:.2f}</td>
                            <th>Daily Change</th>
                            <td class="{day_change_class}">{day_change:.2f}%</td>
                        </tr>
                        <tr>
                            <th>Weekly Change</th>
                            <td class="{self._get_change_class(price_changes.get('weekly', 0))}">{price_changes.get('weekly', 0):.2f}%</td>
                            <th>Monthly Change</th>
                            <td class="{self._get_change_class(price_changes.get('monthly', 0))}">{price_changes.get('monthly', 0):.2f}%</td>
                        </tr>
                        <tr>
                            <th>Yearly Change</th>
                            <td class="{self._get_change_class(price_changes.get('yearly', 0))}">{price_changes.get('yearly', 0):.2f}%</td>
                            <th>Sector</th>
                            <td>{overview.get('company_info', {}).get('sector', 'Unknown')}</td>
                        </tr>
                    </table>
                    
                    <div class="chart">
                        <img src="data:image/png;base64,{report.get('charts', {}).get('price_chart', '')}" alt="Price Chart">
                    </div>
                </div>
                """
            
            # 2. Technical Analysis Section
            if "technical_analysis" in sections:
                technical = sections["technical_analysis"]
                
                html += f"""
                <div class="section">
                    <h2>Technical Analysis</h2>
                    <p>{technical.get("summary", "")}</p>
                    
                    <h3>Key Indicators</h3>
                    <table>
                """
                
                # Add indicators table
                indicators = technical.get("indicators", {})
                if indicators:
                    for name, value in indicators.items():
                        html += f"""
                        <tr>
                            <th>{name.upper()}</th>
                            <td>{value:.2f if isinstance(value, (int, float)) else value}</td>
                        </tr>
                        """
                
                html += """
                    </table>
                    
                    <h3>Trend Analysis</h3>
                    <table>
                """
                
                # Add trend analysis
                trend = technical.get("trend_analysis", {})
                if trend:
                    for name, value in trend.items():
                        html += f"""
                        <tr>
                            <th>{name.replace('_', ' ').title()}</th>
                            <td class="{value if isinstance(value, str) else ''}">{value}</td>
                        </tr>
                        """
                
                html += """
                    </table>
                    
                    <div class="chart">
                        <img src="data:image/png;base64,{}" alt="Technical Indicators Chart">
                    </div>
                </div>
                """.format(report.get('charts', {}).get('indicators_chart', ''))
            
            # 3. Fundamental Analysis Section
            if "fundamental_analysis" in sections:
                fundamental = sections["fundamental_analysis"]
                
                if not isinstance(fundamental, dict) or "status" in fundamental and fundamental["status"] == "error":
                    html += f"""
                    <div class="section">
                        <h2>Fundamental Analysis</h2>
                        <p>Fundamental data not available or insufficient for analysis.</p>
                    </div>
                    """
                else:
                    html += f"""
                    <div class="section">
                        <h2>Fundamental Analysis</h2>
                        <p>{fundamental.get("summary", "")}</p>
                        
                        <h3>Key Financial Metrics</h3>
                        <table>
                    """
                    
                    # Add financial metrics
                    metrics = fundamental.get("financial_metrics", {})
                    if metrics:
                        for name, value in metrics.items():
                            html += f"""
                            <tr>
                                <th>{name.replace('_', ' ').title()}</th>
                                <td>{value:.2f if isinstance(value, (int, float)) else value}</td>
                            </tr>
                            """
                    
                    html += """
                        </table>
                        
                        <h3>Valuation Metrics</h3>
                        <table>
                    """
                    
                    # Add valuation metrics
                    valuation = fundamental.get("valuation_metrics", {})
                    if valuation:
                        for name, value in valuation.items():
                            html += f"""
                            <tr>
                                <th>{name.replace('_', ' ').title()}</th>
                                <td>{value:.2f if isinstance(value, (int, float)) else value}</td>
                            </tr>
                            """
                    
                    html += """
                        </table>
                    </div>
                    """
            
            # 4. Volatility Analysis Section
            if "volatility_analysis" in sections:
                volatility = sections["volatility_analysis"]
                
                html += f"""
                <div class="section">
                    <h2>Volatility Analysis</h2>
                    <p>{volatility.get("summary", "")}</p>
                    
                    <h3>Volatility Metrics</h3>
                    <table>
                        <tr>
                            <th>Current Volatility</th>
                            <td>{volatility.get("current_volatility", 0):.2f}%</td>
                            <th>Historical Volatility</th>
                            <td>{volatility.get("historical_volatility", 0):.2f}%</td>
                        </tr>
                        <tr>
                            <th>Volatility State</th>
                            <td>{volatility.get("volatility_state", "Unknown")}</td>
                            <th>ATR</th>
                            <td>{volatility.get("atr", 0):.2f} ({volatility.get("atr_percent", 0):.2f}% of price)</td>
                        </tr>
                    </table>
                    
                    <div class="chart">
                        <img src="data:image/png;base64,{report.get('charts', {}).get('volatility_chart', '')}" alt="Volatility Chart">
                    </div>
                </div>
                """
            
            # 5. Price Levels Section
            if "price_levels" in sections:
                price_levels = sections["price_levels"]
                
                html += f"""
                <div class="section">
                    <h2>Price Levels</h2>
                    <p>{price_levels.get("summary", "")}</p>
                    
                    <h3>Support and Resistance Levels</h3>
                    <table>
                        <tr>
                            <th>Type</th>
                            <th>Price</th>
                            <th>Strength</th>
                            <th>Distance from Current</th>
                        </tr>
                """
                
                # Add support levels
                sr_levels = price_levels.get("support_resistance", {})
                current_price = price_levels.get("current_price", 0)
                
                for level in sr_levels.get("support_levels", []):
                    price = level.get("price", 0)
                    strength = level.get("strength", 0)
                    distance = abs((price / current_price - 1) * 100)
                    
                    html += f"""
                    <tr>
                        <td>Support</td>
                        <td>{price:.2f}</td>
                        <td>{strength}</td>
                        <td>{distance:.2f}%</td>
                    </tr>
                    """
                
                # Add resistance levels
                for level in sr_levels.get("resistance_levels", []):
                    price = level.get("price", 0)
                    strength = level.get("strength", 0)
                    distance = abs((price / current_price - 1) * 100)
                    
                    html += f"""
                    <tr>
                        <td>Resistance</td>
                        <td>{price:.2f}</td>
                        <td>{strength}</td>
                        <td>{distance:.2f}%</td>
                    </tr>
                    """
                
                html += """
                    </table>
                    
                    <h3>Other Price Levels</h3>
                    <table>
                """
                
                # Add moving average levels
                ma_levels = price_levels.get("moving_averages", {})
                for name, value in ma_levels.items():
                    relation = "Above Price" if value > current_price else "Below Price"
                    distance = abs((value / current_price - 1) * 100)
                    
                    html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{value:.2f}</td>
                        <td>{relation}</td>
                        <td>{distance:.2f}%</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 6. Trading Opportunities Section
            if "opportunities" in sections:
                opportunities = sections["opportunities"]
                
                html += f"""
                <div class="section">
                    <h2>Trading Opportunities</h2>
                    <p>{opportunities.get("summary", "")}</p>
                    
                    <h3>Potential Setups</h3>
                    <table>
                        <tr>
                            <th>Type</th>
                            <th>Direction</th>
                            <th>Score</th>
                            <th>Description</th>
                        </tr>
                """
                
                # Add opportunities
                for opp in opportunities.get("opportunities", []):
                    opp_type = opp.get("type", "Unknown")
                    direction = opp.get("direction", "neutral")
                    score = opp.get("score", 0)
                    description = opp.get("description", "")
                    
                    html += f"""
                    <tr>
                        <td>{opp_type}</td>
                        <td class="{direction}">{direction.capitalize()}</td>
                        <td>{score}</td>
                        <td>{description}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 7. Correlation Analysis Section
            if "correlation_analysis" in sections:
                correlation = sections["correlation_analysis"]
                
                if not isinstance(correlation, dict) or "status" in correlation and correlation["status"] != "success":
                    html += f"""
                    <div class="section">
                        <h2>Correlation Analysis</h2>
                        <p>Correlation data not available or insufficient for analysis.</p>
                    </div>
                    """
                else:
                    html += f"""
                    <div class="section">
                        <h2>Correlation Analysis</h2>
                        <p>{correlation.get("summary", "")}</p>
                        
                        <h3>Top Correlations</h3>
                        <table>
                            <tr>
                                <th>Symbol</th>
                                <th>Correlation</th>
                            </tr>
                    """
                    
                    # Add correlations
                    symbol_correlations = correlation.get("symbol_correlations", [])
                    symbol_correlations.sort(key=lambda x: abs(x.get("correlation", 0)), reverse=True)
                    
                    for corr in symbol_correlations[:5]:  # Top 5 correlations
                        symbol = corr.get("symbol", "Unknown")
                        correlation_value = corr.get("correlation", 0)
                        corr_class = self._get_correlation_class(correlation_value)
                        
                        html += f"""
                        <tr>
                            <td>{symbol}</td>
                            <td class="{corr_class}">{correlation_value:.2f}</td>
                        </tr>
                        """
                    
                    html += """
                        </table>
                    </div>
                    """
            
            # 8. Summary and Recommendation Section
            if "summary_and_recommendation" in sections:
                summary = sections["summary_and_recommendation"]
                
                html += f"""
                <div class="section">
                    <h2>Summary and Recommendation</h2>
                    
                    <div class="summary-box">
                        <p>{summary.get("summary", "")}</p>
                    </div>
                    
                    <h3>Sentiment Breakdown</h3>
                    <table>
                """
                
                # Add sentiment breakdown
                sentiment_breakdown = summary.get("sentiment_breakdown", {})
                for name, value in sentiment_breakdown.items():
                    html += f"""
                    <tr>
                        <th>{name.replace('_', ' ').title()}</th>
                        <td class="{value}">{value.capitalize()}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                    
                    <h3>Trading Recommendations</h3>
                    <table>
                """
                
                # Add timeframe recommendation
                timeframe = summary.get("timeframe_recommendation", {})
                html += f"""
                <tr>
                    <th>Recommended Timeframe</th>
                    <td>{timeframe.get("timeframe", "Unknown")}</td>
                </tr>
                <tr>
                    <th>Reasoning</th>
                    <td>{timeframe.get("reasoning", "")}</td>
                </tr>
                """
                
                # Add strategy recommendation
                strategy = summary.get("strategy_recommendation", {})
                html += f"""
                <tr>
                    <th>Strategy Type</th>
                    <td>{strategy.get("strategy_type", "Unknown")}</td>
                </tr>
                <tr>
                    <th>Entry Strategy</th>
                    <td>{strategy.get("entry_strategy", "")}</td>
                </tr>
                <tr>
                    <th>Exit Strategy</th>
                    <td>{strategy.get("exit_strategy", "")}</td>
                </tr>
                """
                
                html += """
                    </table>
                    
                    <h3>Risk Assessment</h3>
                    <table>
                """
                
                # Add risk assessment
                risk = summary.get("risk_assessment", {})
                html += f"""
                <tr>
                    <th>Risk Level</th>
                    <td>{risk.get("risk_level", "Unknown").capitalize()}</td>
                </tr>
                """
                
                # Add key risks
                key_risks = risk.get("key_risks", [])
                if key_risks:
                    html += f"""
                    <tr>
                        <th>Key Risks</th>
                        <td>{", ".join(key_risks)}</td>
                    </tr>
                    """
                
                # Add risk mitigation
                risk_mitigation = risk.get("risk_mitigation", [])
                if risk_mitigation:
                    html += f"""
                    <tr>
                        <th>Risk Mitigation</th>
                        <td>{", ".join(risk_mitigation)}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # Close HTML document
            html += """
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error formatting stock report: {e}")
            return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"
    
    def _get_change_class(self, change: float) -> str:
        """Get CSS class for price change."""
        if change > 0:
            return "bullish"
        elif change < 0:
            return "bearish"
        return "neutral"
    
    def _get_correlation_class(self, correlation: float) -> str:
        """Get CSS class for correlation value."""
        if correlation > 0.7:
            return "bullish"
        elif correlation < -0.7:
            return "bearish"
        return "neutral"
    
    def _get_market_data(self, symbol: str, exchange: str, 
                      days: int = 252, timeframe: str = "day") -> List[Dict[str, Any]]:
        """
        Get market data from database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            days: Number of days to retrieve
            timeframe: Data timeframe
            
        Returns:
            List of market data documents
        """
        try:
            # Calculate the start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Create query
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # Get data from database
            cursor = self.db.market_data_collection.find(query).sort("timestamp", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return []
    
    def _generate_market_summary(self, indices: List[str], exchange: str) -> Dict[str, Any]:
        """
        Generate market summary section.
        
        Args:
            indices: List of indices
            exchange: Stock exchange
            
        Returns:
            Dictionary with market summary
        """
        try:
            summary = {
                "indices": [],
                "overall_sentiment": "neutral",
                "market_breadth": {},
                "sector_movers": [],
                "summary": ""
            }
            
            # Get data for each index
            for idx in indices:
                data = self._get_market_data(idx, exchange, days=30)
                
                if not data or len(data) < 2:
                    continue
                
                df = pd.DataFrame(data).sort_values("timestamp")
                
                latest_close = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-2]
                day_change = (latest_close / prev_close - 1) * 100
                
                week_ago = df['close'].iloc[-6] if len(df) > 5 else None
                week_change = (latest_close / week_ago - 1) * 100 if week_ago else None
                
                month_ago = df['close'].iloc[-22] if len(df) > 21 else None
                month_change = (latest_close / month_ago - 1) * 100 if month_ago else None
                
                summary["indices"].append({
                    "name": idx,
                    "current_value": latest_close,
                    "day_change": day_change,
                    "week_change": week_change,
                    "month_change": month_change
                })
            
            # Determine overall market sentiment
            if summary["indices"]:
                day_changes = [idx["day_change"] for idx in summary["indices"]]
                avg_day_change = sum(day_changes) / len(day_changes)
                
                if avg_day_change > 1.0:
                    summary["overall_sentiment"] = "bullish"
                elif avg_day_change < -1.0:
                    summary["overall_sentiment"] = "bearish"
                elif avg_day_change > 0.2:
                    summary["overall_sentiment"] = "slightly_bullish"
                elif avg_day_change < -0.2:
                    summary["overall_sentiment"] = "slightly_bearish"
            
            # Get market breadth
            breadth = self._get_market_breadth_data(exchange)
            if breadth:
                summary["market_breadth"] = breadth
            
            # Get sector performance
            sectors = self._get_sector_performance_data(exchange)
            if sectors:
                # Find top and bottom sectors
                sectors.sort(key=lambda x: x.get("day_change", 0), reverse=True)
                top_sectors = sectors[:3]
                bottom_sectors = sectors[-3:]
                
                for sector in top_sectors:
                    summary["sector_movers"].append({
                        "name": sector["name"],
                        "change": sector["day_change"],
                        "direction": "up"
                    })
                
                for sector in bottom_sectors:
                    summary["sector_movers"].append({
                        "name": sector["name"],
                        "change": sector["day_change"],
                        "direction": "down"
                    })
            
            # Generate summary text
            summary_parts = []
            
            if summary["indices"]:
                nifty = next((idx for idx in summary["indices"] if "NIFTY" in idx["name"]), summary["indices"][0])
                summary_parts.append(f"The {nifty['name']} is currently at {nifty['current_value']:.2f}, ")
                
                if nifty["day_change"] > 0:
                    summary_parts.append(f"up {nifty['day_change']:.2f}% for the day. ")
                else:
                    summary_parts.append(f"down {abs(nifty['day_change']):.2f}% for the day. ")
            
            # Add market breadth to summary
            breadth_data = summary["market_breadth"]
            if breadth_data:
                advance_decline = breadth_data.get("advance_decline_ratio", 0)
                
                if advance_decline > 1.5:
                    summary_parts.append(f"Market breadth is positive with an advance-decline ratio of {advance_decline:.2f}. ")
                elif advance_decline < 0.67:
                    summary_parts.append(f"Market breadth is negative with an advance-decline ratio of {advance_decline:.2f}. ")
                else:
                    summary_parts.append(f"Market breadth is mixed with an advance-decline ratio of {advance_decline:.2f}. ")
            
            # Add sector information
            if summary["sector_movers"]:
                top_sector = next((s for s in summary["sector_movers"] if s["direction"] == "up"), None)
                bottom_sector = next((s for s in summary["sector_movers"] if s["direction"] == "down"), None)
                
                if top_sector and bottom_sector:
                    summary_parts.append(f"The {top_sector['name']} sector is leading with a {top_sector['change']:.2f}% gain, while {bottom_sector['name']} is lagging with a {bottom_sector['change']:.2f}% change. ")
            
            # Add overall market sentiment
            sentiment = summary["overall_sentiment"]
            if sentiment == "bullish":
                summary_parts.append("Overall market sentiment is bullish with broad-based buying across sectors.")
            elif sentiment == "slightly_bullish":
                summary_parts.append("Market sentiment is cautiously optimistic with selective buying in key sectors.")
            elif sentiment == "bearish":
                summary_parts.append("Overall market sentiment is bearish with selling pressure across sectors.")
            elif sentiment == "slightly_bearish":
                summary_parts.append("Market sentiment is cautious with some selling pressure.")
            else:
                summary_parts.append("Market sentiment is neutral with mixed signals across sectors.")
            
            summary["summary"] = "".join(summary_parts)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating market summary: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_market_breadth_data(self, exchange: str) -> Dict[str, Any]:
        """
        Get market breadth data.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            Dictionary with market breadth data
        """
        try:
            # Query the market breadth collection
            breadth = self.db.market_breadth_collection.find_one(
                {"exchange": exchange},
                sort=[("timestamp", -1)]
            )
            
            if not breadth:
                return {}
            
            return {
                "advances": breadth.get("advances", 0),
                "declines": breadth.get("declines", 0),
                "unchanged": breadth.get("unchanged", 0),
                "advance_decline_ratio": breadth.get("advance_decline_ratio", 0),
                "new_highs": breadth.get("new_highs", 0),
                "new_lows": breadth.get("new_lows", 0),
                "timestamp": breadth.get("timestamp", datetime.now())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market breadth data: {e}")
            return {}
    
    def _get_sector_performance_data(self, exchange: str) -> List[Dict[str, Any]]:
        """
        Get sector performance data.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            List of sector performance data
        """
        try:
            # Query the sector performance collection
            cursor = self.db.sector_performance_collection.find(
                {"exchange": exchange},
                sort=[("timestamp", -1)]
            ).limit(1)
            
            sector_data = list(cursor)
            
            if not sector_data:
                return []
            
            sectors = sector_data[0].get("sectors", [])
            
            return sectors
            
        except Exception as e:
            self.logger.error(f"Error getting sector performance data: {e}")
            return []
    
    def _generate_index_performance(self, indices: List[str], exchange: str) -> Dict[str, Any]:
        """
        Generate index performance section.
        
        Args:
            indices: List of indices
            exchange: Stock exchange
            
        Returns:
            Dictionary with index performance
        """
        try:
            performance = {
                "indices": [],
                "charts": {},
                "summary": ""
            }
            
            # Get data for each index
            for idx in indices:
                data = self._get_market_data(idx, exchange, days=252)
                
                if not data or len(data) < 30:
                    continue
                
                df = pd.DataFrame(data).sort_values("timestamp")
                
                latest_close = df['close'].iloc[-1]
                
                # Calculate various timeframe returns
                returns = {}
                
                if len(df) >= 2:
                    returns["day"] = (latest_close / df['close'].iloc[-2] - 1) * 100
                
                if len(df) >= 6:
                    returns["week"] = (latest_close / df['close'].iloc[-6] - 1) * 100
                
                if len(df) >= 22:
                    returns["month"] = (latest_close / df['close'].iloc[-22] - 1) * 100
                
                if len(df) >= 63:
                    returns["quarter"] = (latest_close / df['close'].iloc[-63] - 1) * 100
                
                if len(df) >= 126:
                    returns["half_year"] = (latest_close / df['close'].iloc[-126] - 1) * 100
                
                if len(df) >= 252:
                    returns["year"] = (latest_close / df['close'].iloc[-252] - 1) * 100
                
                # Calculate moving averages
                ma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
                ma_200 = df['close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
                
                # Calculate 52-week range
                high_52w = df['high'].max()
                low_52w = df['low'].min()
                
                # Calculate distance from 52-week high/low
                high_dist = (latest_close / high_52w - 1) * 100
                low_dist = (latest_close / low_52w - 1) * 100
                
                # Add to performance data
                performance["indices"].append({
                    "name": idx,
                    "current_value": latest_close,
                    "returns": returns,
                    "moving_averages": {
                        "ma_50": ma_50,
                        "ma_200": ma_200,
                        "vs_ma50": ((latest_close / ma_50) - 1) * 100 if ma_50 else None,
                        "vs_ma200": ((latest_close / ma_200) - 1) * 100 if ma_200 else None
                    },
                    "range_52w": {
                        "high": high_52w,
                        "low": low_52w,
                        "high_distance": high_dist,
                        "low_distance": low_dist
                    }
                })
                
                # Create chart for this index
                chart = self._create_index_chart(df, idx)
                if chart:
                    performance["charts"][idx] = chart
            
            # Generate summary text
            if performance["indices"]:
                summary_parts = []
                
                # Compare index performances
                indices_sorted = sorted(performance["indices"], key=lambda x: x["returns"].get("month", 0), reverse=True)
                
                best_performer = indices_sorted[0]
                worst_performer = indices_sorted[-1]
                
                month_return_best = best_performer["returns"].get("month")
                month_return_worst = worst_performer["returns"].get("month")
                
                if month_return_best is not None and month_return_worst is not None:
                    summary_parts.append(f"Over the past month, {best_performer['name']} has been the best performing index with a {month_return_best:.2f}% return, while {worst_performer['name']} has been the worst with a {month_return_worst:.2f}% return. ")
                
                # Check for golden/death crosses
                for idx in performance["indices"]:
                    ma_data = idx["moving_averages"]
                    ma_50 = ma_data.get("ma_50")
                    ma_200 = ma_data.get("ma_200")
                    
                    if ma_50 is not None and ma_200 is not None:
                        if ma_50 > ma_200 and ma_data.get("vs_ma50", 0) > 0 and ma_data.get("vs_ma200", 0) > 0:
                            summary_parts.append(f"{idx['name']} is in a bullish configuration with price above both 50-day and 200-day moving averages, and the 50-day average above the 200-day average. ")
                        elif ma_50 < ma_200 and ma_data.get("vs_ma50", 0) < 0 and ma_data.get("vs_ma200", 0) < 0:
                            summary_parts.append(f"{idx['name']} is in a bearish configuration with price below both 50-day and 200-day moving averages, and the 50-day average below the 200-day average. ")
                
                # Overall market direction
                avg_month_return = sum(idx["returns"].get("month", 0) for idx in performance["indices"] if "month" in idx["returns"]) / len(performance["indices"])
                
                if avg_month_return > 3:
                    summary_parts.append("Overall, the market has been strongly bullish over the past month.")
                elif avg_month_return > 1:
                    summary_parts.append("Overall, the market has been moderately bullish over the past month.")
                elif avg_month_return > -1:
                    summary_parts.append("Overall, the market has been range-bound over the past month.")
                elif avg_month_return > -3:
                    summary_parts.append("Overall, the market has been moderately bearish over the past month.")
                else:
                    summary_parts.append("Overall, the market has been strongly bearish over the past month.")
                
                performance["summary"] = " ".join(summary_parts)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error generating index performance: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_index_chart(self, df: pd.DataFrame, index_name: str) -> str:
        """
        Create index performance chart.
        
        Args:
            df: DataFrame with market data
            index_name: Index name
            
        Returns:
            Base64 encoded chart image
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot index price
            ax.plot(df['timestamp'], df['close'], label='Close Price')
            
            # Add moving averages
            if len(df) >= 50:
                ma50 = df['close'].rolling(window=50).mean()
                ax.plot(df['timestamp'], ma50, label='50-day MA')
            
            if len(df) >= 200:
                ma200 = df['close'].rolling(window=200).mean()
                ax.plot(df['timestamp'], ma200, label='200-day MA')
            
            ax.set_title(f'{index_name} Performance')
            ax.set_xlabel('Date')
            ax.set_ylabel('Index Value')
            ax.legend()
            ax.grid(True)
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error creating index chart: {e}")
            return ""
    
    def _generate_sector_performance(self, exchange: str) -> Dict[str, Any]:
        """
        Generate sector performance section.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            Dictionary with sector performance
        """
        try:
            performance = {
                "sectors": [],
                "chart": "",
                "summary": ""
            }
            
            # Get sector performance data
            sectors = self._get_sector_performance_data(exchange)
            
            if not sectors:
                return {
                    "status": "not_available",
                    "message": "Sector performance data not available"
                }
            
            performance["sectors"] = sectors
            
            # Create sector performance chart
            performance["chart"] = self._create_sector_chart(sectors)
            
            # Generate summary text
            sectors_sorted = sorted(sectors, key=lambda x: x.get("day_change", 0), reverse=True)
            
            if sectors_sorted:
                summary_parts = []
                
                top_sectors = sectors_sorted[:3]
                bottom_sectors = sectors_sorted[-3:]
                
                # Top performing sectors
                top_sectors_str = ", ".join(f"{s['name']} ({s['day_change']:.2f}%)" for s in top_sectors)
                summary_parts.append(f"The top performing sectors today are {top_sectors_str}. ")
                
                # Bottom performing sectors
                bottom_sectors_str = ", ".join(f"{s['name']} ({s['day_change']:.2f}%)" for s in bottom_sectors)
                summary_parts.append(f"The worst performing sectors are {bottom_sectors_str}. ")
                
                # Count positive sectors
                positive_sectors = sum(1 for s in sectors if s.get("day_change", 0) > 0)
                total_sectors = len(sectors)
                
                if positive_sectors > total_sectors * 0.7:
                    summary_parts.append(f"Market breadth is strong with {positive_sectors} out of {total_sectors} sectors showing positive returns. ")
                elif positive_sectors < total_sectors * 0.3:
                    summary_parts.append(f"Market breadth is weak with only {positive_sectors} out of {total_sectors} sectors showing positive returns. ")
                else:
                    summary_parts.append(f"Market breadth is mixed with {positive_sectors} out of {total_sectors} sectors showing positive returns. ")
                
                # Sector rotation analysis
                if len(sectors_sorted) >= 6:
                    defensive_sectors = ["FMCG", "Healthcare", "Pharma", "Consumer", "Utilities"]
                    cyclical_sectors = ["Auto", "Metal", "Real Estate", "Construction", "Banking"]
                    tech_sectors = ["IT", "Technology"]
                    
                    # Check if defensive or cyclical sectors are leading
                    top_6_names = [s["name"] for s in sectors_sorted[:6]]
                    
                    defensive_count = sum(1 for name in top_6_names if any(d in name for d in defensive_sectors))
                    cyclical_count = sum(1 for name in top_6_names if any(c in name for c in cyclical_sectors))
                    tech_count = sum(1 for name in top_6_names if any(t in name for t in tech_sectors))
                    
                    if defensive_count >= 3:
                        summary_parts.append("Defensive sectors are leading, which may indicate cautious market sentiment. ")
                    elif cyclical_count >= 3:
                        summary_parts.append("Cyclical sectors are leading, which may indicate optimistic economic outlook. ")
                    elif tech_count >= 2:
                        summary_parts.append("Technology sectors are showing strength. ")
                
                performance["summary"] = "".join(summary_parts)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error generating sector performance: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_sector_chart(self, sectors: List[Dict[str, Any]]) -> str:
        """
        Create sector performance chart.
        
        Args:
            sectors: List of sector data
            
        Returns:
            Base64 encoded chart image
        """
        try:
            # Sort sectors by performance
            sectors_sorted = sorted(sectors, key=lambda x: x.get("day_change", 0))
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            sector_names = [s["name"] for s in sectors_sorted]
            returns = [s.get("day_change", 0) for s in sectors_sorted]
            
            # Set colors based on returns
            colors = ['red' if ret < 0 else 'green' for ret in returns]
            
            # Create horizontal bar chart
            bars = ax.barh(sector_names, returns, color=colors)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
                        ha='left', va='center')
            
            ax.set_title('Sector Performance (%)')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(axis='x', alpha=0.3)
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error creating sector chart: {e}")
            return ""
    
    def _generate_market_breadth(self, exchange: str) -> Dict[str, Any]:
        """
        Generate market breadth section.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            Dictionary with market breadth
        """
        try:
            # Get market breadth data
            current_breadth = self._get_market_breadth_data(exchange)
            
            if not current_breadth:
                return {
                    "status": "not_available",
                    "message": "Market breadth data not available"
                }
            
            # Get historical breadth data
            historical_breadth = self._get_historical_breadth_data(exchange)
            
            # Calculate advance-decline line
            adv_dec_line = self._calculate_adv_dec_line(historical_breadth)
            
            # Generate summary text
            summary = self._generate_breadth_summary(current_breadth, historical_breadth)
            
            return {
                "current_breadth": current_breadth,
                "historical_data": historical_breadth[-10:] if historical_breadth else [],
                "advance_decline_line": adv_dec_line,
                "chart": self._create_breadth_chart(historical_breadth),
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating market breadth: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_historical_breadth_data(self, exchange: str) -> List[Dict[str, Any]]:
        """
        Get historical market breadth data.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            List of historical market breadth data
        """
        try:
            # Query the market breadth collection
            cursor = self.db.market_breadth_collection.find(
                {"exchange": exchange},
                sort=[("timestamp", 1)]
            ).limit(30)  # Last 30 days
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting historical breadth data: {e}")
            return []
    
    def _calculate_adv_dec_line(self, historical_breadth: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate advance-decline line.
        
        Args:
            historical_breadth: Historical market breadth data
            
        Returns:
            List of advance-decline line data
        """
        try:
            if not historical_breadth:
                return []
            
            adv_dec_line = []
            cumulative_value = 0
            
            for day in historical_breadth:
                advances = day.get("advances", 0)
                declines = day.get("declines", 0)
                
                # Calculate daily value (advances - declines)
                daily_value = advances - declines
                
                # Add to cumulative value
                cumulative_value += daily_value
                
                adv_dec_line.append({
                    "timestamp": day.get("timestamp"),
                    "value": cumulative_value
                })
            
            return adv_dec_line
            
        except Exception as e:
            self.logger.error(f"Error calculating advance-decline line: {e}")
            return []
    
    def _create_breadth_chart(self, historical_breadth: List[Dict[str, Any]]) -> str:
        """
        Create market breadth chart.
        
        Args:
            historical_breadth: Historical market breadth data
            
        Returns:
            Base64 encoded chart image
        """
        try:
            if not historical_breadth:
                return ""
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
            
            # Extract data
            dates = [day.get("timestamp") for day in historical_breadth]
            advances = [day.get("advances", 0) for day in historical_breadth]
            declines = [day.get("declines", 0) for day in historical_breadth]
            unchanged = [day.get("unchanged", 0) for day in historical_breadth]
            
            # Calculate advance-decline line
            adv_dec_line = []
            cumulative_value = 0
            
            for i, day in enumerate(historical_breadth):
                daily_value = advances[i] - declines[i]
                cumulative_value += daily_value
                adv_dec_line.append(cumulative_value)
            
            # Plot advances-declines
            width = 0.35
            x = np.arange(len(dates))
            
            ax1.bar(x, advances, width, label='Advances', color='green', alpha=0.7)
            ax1.bar(x, declines, width, bottom=advances, label='Declines', color='red', alpha=0.7)
            ax1.bar(x, unchanged, width, bottom=[a+d for a, d in zip(advances, declines)], label='Unchanged', color='gray', alpha=0.7)
            
            ax1.set_title('Market Breadth (Advances/Declines)')
            ax1.set_ylabel('Number of Stocks')
            ax1.set_xticks(x)
            ax1.set_xticklabels([d.strftime('%Y-%m-%d') if isinstance(d, datetime) else str(d) for d in dates], rotation=45)
            ax1.legend()
            
            # Plot advance-decline line
            ax2.plot(dates, adv_dec_line, color='blue', marker='o', linestyle='-')
            ax2.set_title('Advance-Decline Line')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative A-D')
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error creating breadth chart: {e}")
            return ""
    
    def _generate_breadth_summary(self, current: Dict[str, Any], 
                               historical: List[Dict[str, Any]]) -> str:
        """
        Generate market breadth summary.
        
        Args:
            current: Current market breadth data
            historical: Historical market breadth data
            
        Returns:
            Summary string
        """
        try:
            summary_parts = []
            
            advances = current.get("advances", 0)
            declines = current.get("declines", 0)
            unchanged = current.get("unchanged", 0)
            total = advances + declines + unchanged
            
            if total > 0:
                advance_pct = advances / total * 100
                decline_pct = declines / total * 100
                
                summary_parts.append(f"Today's market breadth shows {advances} advances ({advance_pct:.1f}%) and {declines} declines ({decline_pct:.1f}%).")
            
            ad_ratio = current.get("advance_decline_ratio", 0)
            
            if ad_ratio > 2:
                summary_parts.append(f"The advance-decline ratio of {ad_ratio:.2f} indicates very strong market breadth.")
            elif ad_ratio > 1.5:
                summary_parts.append(f"The advance-decline ratio of {ad_ratio:.2f} indicates strong market breadth.")
            elif ad_ratio > 1:
                summary_parts.append(f"The advance-decline ratio of {ad_ratio:.2f} indicates moderately positive market breadth.")
            elif ad_ratio > 0.67:
                summary_parts.append(f"The advance-decline ratio of {ad_ratio:.2f} indicates neutral market breadth.")
            elif ad_ratio > 0.5:
                summary_parts.append(f"The advance-decline ratio of {ad_ratio:.2f} indicates moderately negative market breadth.")
            else:
                summary_parts.append(f"The advance-decline ratio of {ad_ratio:.2f} indicates weak market breadth.")
            
            # Analyze historical trend if available
            if len(historical) > 5:
                recent_advances = [day.get("advances", 0) for day in historical[-5:]]
                recent_declines = [day.get("declines", 0) for day in historical[-5:]]
                
                avg_advances = sum(recent_advances) / len(recent_advances)
                avg_declines = sum(recent_declines) / len(recent_declines)
                
                if avg_advances > avg_declines * 1.5:
                    summary_parts.append("The market has shown consistently strong breadth over the past week, indicating broad participation in the market move.")
                elif avg_advances < avg_declines * 0.67:
                    summary_parts.append("The market has shown consistently weak breadth over the past week, indicating broad-based selling pressure.")
            
            # New highs vs. new lows
            new_highs = current.get("new_highs", 0)
            new_lows = current.get("new_lows", 0)
            
            if new_highs > 0 or new_lows > 0:
                summary_parts.append(f"There are {new_highs} stocks making new 52-week highs and {new_lows} making new 52-week lows.")
                
                if new_highs > new_lows * 3:
                    summary_parts.append("The high number of stocks making new highs is a bullish indicator.")
                elif new_lows > new_highs * 3:
                    summary_parts.append("The high number of stocks making new lows is a bearish indicator.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating breadth summary: {e}")
            return "Market breadth data analysis not available."
    
    def _generate_volatility_overview(self, indices: List[str], exchange: str) -> Dict[str, Any]:
        """
        Generate volatility overview section.
        
        Args:
            indices: List of indices
            exchange: Stock exchange
            
        Returns:
            Dictionary with volatility overview
        """
        try:
            overview = {
                "index_volatility": [],
                "vix_data": [],
                "volatility_regime": "normal",
                "chart": "",
                "summary": ""
            }
            
            # Get volatility data for each index
            for idx in indices:
                data = self._get_market_data(idx, exchange, days=63)
                
                if not data or len(data) < 21:
                    continue
                
                df = pd.DataFrame(data).sort_values("timestamp")
                
                # Calculate historical volatility
                df['returns'] = df['close'].pct_change()
                current_vol = df['returns'].rolling(window=21).std().iloc[-1] * 100 * math.sqrt(252)  # Annualized
                
                # Calculate historical volatility for comparison
                if len(df) >= 63:
                    historical_vol = df['returns'].rolling(window=21).std().iloc[-63:-21].mean() * 100 * math.sqrt(252)
                    vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
                else:
                    historical_vol = None
                    vol_ratio = 1
                
                # Determine volatility state
                vol_state = "stable"
                if vol_ratio > 1.5:
                    vol_state = "expanding"
                elif vol_ratio < 0.7:
                    vol_state = "contracting"
                
                overview["index_volatility"].append({
                    "index": idx,
                    "current_volatility": current_vol,
                    "historical_volatility": historical_vol,
                    "volatility_ratio": vol_ratio,
                    "volatility_state": vol_state
                })
            
            # Get VIX data if available
            vix_data = self._get_market_data("VIX", exchange, days=30) or self._get_market_data("INDIA VIX", exchange, days=30)
            
            if vix_data:
                df = pd.DataFrame(vix_data).sort_values("timestamp")
                
                current_vix = df['close'].iloc[-1]
                avg_vix = df['close'].mean()
                vix_percentile = stats.percentileofscore(df['close'].values, current_vix)
                
                overview["vix_data"] = {
                    "current_value": current_vix,
                    "average_value": avg_vix,
                    "percentile": vix_percentile
                }
                
                # Determine overall volatility regime
                if vix_percentile > 80:
                    overview["volatility_regime"] = "high"
                elif vix_percentile < 20:
                    overview["volatility_regime"] = "low"
                else:
                    overview["volatility_regime"] = "normal"
                
                # Create VIX chart
                overview["chart"] = self._create_vix_chart(df)
            
            # Generate summary text
            summary_parts = []
            
            # VIX summary
            if "current_value" in overview.get("vix_data", {}):
                vix = overview["vix_data"]["current_value"]
                vix_percentile = overview["vix_data"]["percentile"]
                
                if vix_percentile > 80:
                    summary_parts.append(f"Market volatility is elevated with the VIX at {vix:.2f}, in the {vix_percentile:.0f}th percentile of its recent range.")
                elif vix_percentile < 20:
                    summary_parts.append(f"Market volatility is subdued with the VIX at {vix:.2f}, in the {vix_percentile:.0f}th percentile of its recent range.")
                else:
                    summary_parts.append(f"Market volatility is normal with the VIX at {vix:.2f}, in the {vix_percentile:.0f}th percentile of its recent range.")
            
            # Index volatility summary
            if overview["index_volatility"]:
                expanding_indices = [item["index"] for item in overview["index_volatility"] if item["volatility_state"] == "expanding"]
                contracting_indices = [item["index"] for item in overview["index_volatility"] if item["volatility_state"] == "contracting"]
                
                if expanding_indices:
                    summary_parts.append(f"Volatility is expanding in {', '.join(expanding_indices)}.")
                
                if contracting_indices:
                    summary_parts.append(f"Volatility is contracting in {', '.join(contracting_indices)}.")
            
            # Trading implications
            if overview["volatility_regime"] == "high":
                summary_parts.append("High volatility environment suggests using tighter stops, smaller position sizes, and strategies that benefit from volatility such as range-bound trading or option strategies.")
            elif overview["volatility_regime"] == "low":
                summary_parts.append("Low volatility environment is favorable for trend-following strategies and may precede volatility breakouts, so be alert for developing trends.")
            else:
                summary_parts.append("Normal volatility environment is suitable for a balanced approach to trading with standard position sizing and risk management.")
            
            overview["summary"] = " ".join(summary_parts)
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Error generating volatility overview: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_vix_chart(self, df: pd.DataFrame) -> str:
        """
        Create VIX chart.
        
        Args:
            df: DataFrame with VIX data
            
        Returns:
            Base64 encoded chart image
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot VIX
            ax.plot(df['timestamp'], df['close'], color='purple', linewidth=2)
            
            # Add horizontal lines for different volatility regimes
            avg_vix = df['close'].mean()
            ax.axhline(y=avg_vix, color='gray', linestyle='--', alpha=0.7, label='Average')
            
            # Add high and low volatility thresholds (e.g., 80th and 20th percentiles)
            high_threshold = np.percentile(df['close'].values, 80)
            low_threshold = np.percentile(df['close'].values, 20)
            
            ax.axhline(y=high_threshold, color='red', linestyle='--', alpha=0.7, label='High Volatility')
            ax.axhline(y=low_threshold, color='green', linestyle='--', alpha=0.7, label='Low Volatility')
            
            ax.set_title('VIX (Volatility Index)')
            ax.set_xlabel('Date')
            ax.set_ylabel('VIX Value')
            ax.legend()
            ax.grid(True)
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error creating VIX chart: {e}")
            return ""
    
    def _generate_correlation_overview(self, indices: List[str], exchange: str) -> Dict[str, Any]:
        """
        Generate correlation overview section.
        
        Args:
            indices: List of indices
            exchange: Stock exchange
            
        Returns:
            Dictionary with correlation overview
        """
        try:
            # Use correlation analyzer if available
            if self.correlation_analyzer:
                corr_analysis = self.correlation_analyzer.analyze_correlation_matrix(indices, exchange)
                
                if corr_analysis and "status" in corr_analysis and corr_analysis["status"] == "success":
                    return {
                        "correlation_matrix": corr_analysis.get("correlation_matrix", []),
                        "high_correlation_pairs": corr_analysis.get("high_correlation_pairs", []),
                        "inverse_correlation_pairs": corr_analysis.get("inverse_correlation_pairs", []),
                        "average_correlation": corr_analysis.get("average_correlation", 0),
                        "summary": corr_analysis.get("correlation_summary", "")
                    }
            
            # Fallback to simple correlation analysis
            return self._calculate_simple_correlations(indices, exchange)
            
        except Exception as e:
            self.logger.error(f"Error generating correlation overview: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_simple_correlations(self, indices: List[str], exchange: str) -> Dict[str, Any]:
        """
        Calculate simple correlations between indices.
        
        Args:
            indices: List of indices
            exchange: Stock exchange
            
        Returns:
            Dictionary with correlation data
        """
        try:
            # Get price data for all indices
            price_data = {}
            
            for idx in indices:
                data = self._get_market_data(idx, exchange, days=63)
                
                if not data or len(data) < 30:
                    continue
                
                df = pd.DataFrame(data)
                
                # Extract closing prices with timestamps
                idx_data = df[["timestamp", "close"]].sort_values("timestamp")
                price_data[idx] = idx_data
            
            if len(price_data) < 2:
                return {
                    "status": "insufficient_data",
                    "message": "Insufficient data for correlation analysis"
                }
            
            # Create a combined dataframe with all closing prices
            combined_df = None
            
            for idx, data in price_data.items():
                if combined_df is None:
                    combined_df = data.rename(columns={"close": idx}).set_index("timestamp")
                else:
                    combined_df[idx] = data.set_index("timestamp")["close"]
            
            # Fill missing values
            combined_df = combined_df.fillna(method="ffill").fillna(method="bfill")
            
            # Calculate returns
            returns_df = combined_df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Format correlation matrix for output
            formatted_matrix = []
            valid_indices = list(correlation_matrix.columns)
            
            for idx1 in valid_indices:
                row = {"symbol": idx1}
                for idx2 in valid_indices:
                    row[idx2] = round(correlation_matrix.loc[idx1, idx2], 2)
                formatted_matrix.append(row)
            
            # Find highly correlated and inversely correlated pairs
            high_correlation_pairs = []
            inverse_correlation_pairs = []
            
            for i in range(len(valid_indices)):
                for j in range(i+1, len(valid_indices)):
                    idx1 = valid_indices[i]
                    idx2 = valid_indices[j]
                    corr = correlation_matrix.loc[idx1, idx2]
                    
                    if corr >= 0.7:
                        high_correlation_pairs.append({
                            "symbol1": idx1,
                            "symbol2": idx2,
                            "correlation": corr
                        })
                    elif corr <= -0.7:
                        inverse_correlation_pairs.append({
                            "symbol1": idx1,
                            "symbol2": idx2,
                            "correlation": corr
                        })
            
            # Calculate average correlation
            corr_values = []
            for i in range(len(valid_indices)):
                for j in range(i+1, len(valid_indices)):
                    corr_values.append(correlation_matrix.loc[valid_indices[i], valid_indices[j]])
            
            avg_correlation = sum(corr_values) / len(corr_values) if corr_values else 0
            # Generate summary text
            summary_parts = []
            
            if avg_correlation > 0.7:
                summary_parts.append(f"The analyzed indices show high average correlation ({avg_correlation:.2f}), indicating strong co-movement.")
            elif avg_correlation > 0.4:
                summary_parts.append(f"The analyzed indices show moderate average correlation ({avg_correlation:.2f}).")
            else:
                summary_parts.append(f"The analyzed indices show low average correlation ({avg_correlation:.2f}), indicating good diversification across markets.")
            
            # Mention specific correlations
            if high_correlation_pairs:
                top_pair = high_correlation_pairs[0]
                summary_parts.append(f"The most highly correlated indices are {top_pair['symbol1']} and {top_pair['symbol2']} ({top_pair['correlation']:.2f}).")
            
            if inverse_correlation_pairs:
                top_inverse = inverse_correlation_pairs[0]
                summary_parts.append(f"The most inversely correlated indices are {top_inverse['symbol1']} and {top_inverse['symbol2']} ({top_inverse['correlation']:.2f}).")
            
            # Trading implications
            if avg_correlation > 0.7:
                summary_parts.append("High correlation suggests that diversification across these indices provides limited risk reduction benefits.")
            elif len(inverse_correlation_pairs) > 0:
                summary_parts.append("Inverse correlations provide opportunities for portfolio hedging and risk reduction.")
            
            return {
                "correlation_matrix": formatted_matrix,
                "high_correlation_pairs": high_correlation_pairs,
                "inverse_correlation_pairs": inverse_correlation_pairs,
                "average_correlation": avg_correlation,
                "summary": " ".join(summary_parts)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating simple correlations: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_sentiment_analysis(self, exchange: str) -> Dict[str, Any]:
        """
        Generate sentiment analysis section.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            Dictionary with sentiment analysis
        """
        try:
            # Get sentiment data from database
            sentiment = self._get_sentiment_data(exchange)
            
            if not sentiment:
                return {
                    "status": "not_available",
                    "message": "Sentiment data not available"
                }
            
            # Generate summary text
            summary = self._generate_sentiment_summary(sentiment)
            
            return {
                "news_sentiment": sentiment.get("news_sentiment", {}),
                "social_sentiment": sentiment.get("social_sentiment", {}),
                "technical_sentiment": sentiment.get("technical_sentiment", {}),
                "overall_sentiment": sentiment.get("overall_sentiment", "neutral"),
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment analysis: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_sentiment_data(self, exchange: str) -> Dict[str, Any]:
        """
        Get sentiment data from database.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            Dictionary with sentiment data
        """
        try:
            # Query the sentiment collection
            sentiment = self.db.sentiment_collection.find_one(
                {"exchange": exchange},
                sort=[("timestamp", -1)]
            )
            
            if not sentiment:
                return {}
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment data: {e}")
            return {}
    
    def _generate_sentiment_summary(self, sentiment: Dict[str, Any]) -> str:
        """
        Generate sentiment summary text.
        
        Args:
            sentiment: Sentiment data
            
        Returns:
            Summary string
        """
        try:
            summary_parts = []
            
            # Overall sentiment
            overall = sentiment.get("overall_sentiment", "neutral")
            
            if overall == "bullish":
                summary_parts.append("Overall market sentiment is bullish based on a combination of news, social, and technical indicators.")
            elif overall == "moderately_bullish":
                summary_parts.append("Overall market sentiment is moderately bullish.")
            elif overall == "bearish":
                summary_parts.append("Overall market sentiment is bearish based on a combination of news, social, and technical indicators.")
            elif overall == "moderately_bearish":
                summary_parts.append("Overall market sentiment is moderately bearish.")
            else:
                summary_parts.append("Overall market sentiment is neutral with mixed signals.")
            
            # News sentiment
            news = sentiment.get("news_sentiment", {})
            if news:
                positive_pct = news.get("positive_percent", 0)
                negative_pct = news.get("negative_percent", 0)
                neutral_pct = news.get("neutral_percent", 0)
                
                if positive_pct > negative_pct + 20:
                    summary_parts.append(f"News sentiment is positive with {positive_pct:.0f}% positive coverage.")
                elif negative_pct > positive_pct + 20:
                    summary_parts.append(f"News sentiment is negative with {negative_pct:.0f}% negative coverage.")
                else:
                    summary_parts.append(f"News sentiment is mixed with {positive_pct:.0f}% positive and {negative_pct:.0f}% negative coverage.")
            
            # Social sentiment
            social = sentiment.get("social_sentiment", {})
            if social:
                bullish_pct = social.get("bullish_percent", 0)
                bearish_pct = social.get("bearish_percent", 0)
                
                if bullish_pct > bearish_pct + 20:
                    summary_parts.append(f"Social media sentiment is bullish with {bullish_pct:.0f}% bullish mentions.")
                elif bearish_pct > bullish_pct + 20:
                    summary_parts.append(f"Social media sentiment is bearish with {bearish_pct:.0f}% bearish mentions.")
                else:
                    summary_parts.append(f"Social media sentiment is mixed with {bullish_pct:.0f}% bullish and {bearish_pct:.0f}% bearish mentions.")
            
            # Technical sentiment
            technical = sentiment.get("technical_sentiment", {})
            if technical:
                bullish_indicators = technical.get("bullish_indicators", 0)
                bearish_indicators = technical.get("bearish_indicators", 0)
                total_indicators = bullish_indicators + bearish_indicators
                
                if total_indicators > 0:
                    bullish_pct = bullish_indicators / total_indicators * 100
                    
                    if bullish_pct > 70:
                        summary_parts.append(f"Technical indicators are predominantly bullish ({bullish_pct:.0f}% of indicators).")
                    elif bullish_pct < 30:
                        summary_parts.append(f"Technical indicators are predominantly bearish ({(100-bullish_pct):.0f}% of indicators).")
                    else:
                        summary_parts.append(f"Technical indicators are mixed with {bullish_pct:.0f}% bullish signals.")
            
            # Add trading implications
            if overall == "bullish" or overall == "moderately_bullish":
                summary_parts.append("The positive sentiment suggests a favorable environment for long positions, particularly in sectors with strong momentum.")
            elif overall == "bearish" or overall == "moderately_bearish":
                summary_parts.append("The negative sentiment suggests caution with long positions and potentially opportunities on the short side.")
            else:
                summary_parts.append("The mixed sentiment suggests a selective approach, focusing on individual opportunities rather than broad market direction.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment summary: {e}")
            return "Sentiment analysis not available."
    
    def _get_upcoming_events(self, exchange: str) -> Dict[str, Any]:
        """
        Get upcoming market events.
        
        Args:
            exchange: Stock exchange
            
        Returns:
            Dictionary with upcoming events
        """
        try:
            # Query the events collection
            events = self.db.market_events_collection.find(
                {
                    "exchange": exchange,
                    "date": {"$gte": datetime.now()}
                },
                sort=[("date", 1)]
            ).limit(10)
            
            events_list = list(events)
            
            if not events_list:
                return {
                    "status": "not_available",
                    "message": "Upcoming events data not available"
                }
            
            # Categorize events
            economic_events = []
            earnings_events = []
            other_events = []
            
            for event in events_list:
                event_type = event.get("type", "other")
                
                if event_type == "economic":
                    economic_events.append(event)
                elif event_type == "earnings":
                    earnings_events.append(event)
                else:
                    other_events.append(event)
            
            # Generate summary text
            summary = self._generate_events_summary(economic_events, earnings_events, other_events)
            
            return {
                "economic_events": economic_events,
                "earnings_events": earnings_events,
                "other_events": other_events,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error getting upcoming events: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_events_summary(self, economic_events: List[Dict[str, Any]],
                              earnings_events: List[Dict[str, Any]],
                              other_events: List[Dict[str, Any]]) -> str:
        """
        Generate summary of upcoming events.
        
        Args:
            economic_events: List of economic events
            earnings_events: List of earnings events
            other_events: List of other events
            
        Returns:
            Summary string
        """
        try:
            summary_parts = []
            
            # Economic events
            if economic_events:
                high_impact_events = [event for event in economic_events if event.get("impact", "medium") == "high"]
                
                if high_impact_events:
                    next_event = high_impact_events[0]
                    event_name = next_event.get("name", "Unknown economic event")
                    event_date = next_event.get("date", datetime.now())
                    
                    if isinstance(event_date, datetime):
                        date_str = event_date.strftime("%Y-%m-%d %H:%M")
                    else:
                        date_str = str(event_date)
                    
                    summary_parts.append(f"Key upcoming economic event: {event_name} on {date_str}.")
                
                summary_parts.append(f"There are {len(economic_events)} upcoming economic events that may impact the market.")
            
            # Earnings events
            if earnings_events:
                next_earnings = earnings_events[0]
                company = next_earnings.get("company", "Unknown")
                event_date = next_earnings.get("date", datetime.now())
                
                if isinstance(event_date, datetime):
                    date_str = event_date.strftime("%Y-%m-%d")
                else:
                    date_str = str(event_date)
                
                notable_companies = [event.get("company", "Unknown") for event in earnings_events[:3]]
                
                summary_parts.append(f"Notable upcoming earnings reports include {', '.join(notable_companies)}.")
                summary_parts.append(f"There are {len(earnings_events)} companies scheduled to report earnings in the near future.")
            
            # Other events
            if other_events:
                summary_parts.append(f"There are {len(other_events)} other market events scheduled.")
            
            # Market impact
            if economic_events or earnings_events:
                summary_parts.append("Traders should monitor these events as they may create volatility and trading opportunities.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating events summary: {e}")
            return "Upcoming events information not available."
    
    def _format_market_report(self, report: Dict[str, Any]) -> str:
        """
        Format market report as HTML.
        
        Args:
            report: Report data
            
        Returns:
            HTML formatted report
        """
        try:
            # Get report sections
            sections = report.get("sections", {})
            exchange = report.get("exchange", "")
            generated_at = report.get("generated_at", datetime.now())
            
            # Start building HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Market Overview Report: {exchange}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                    h1, h2, h3, h4 {{ color: #2c3e50; }}
                    .report-header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                    .chart {{ margin: 20px 0; text-align: center; }}
                    .chart img {{ max-width: 100%; height: auto; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .bullish {{ color: green; }}
                    .bearish {{ color: red; }}
                    .neutral {{ color: #888; }}
                    .summary-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1>Market Overview Report: {exchange}</h1>
                    <p>Generated on {generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """
            
            # 1. Market Summary Section
            if "market_summary" in sections:
                market_summary = sections["market_summary"]
                indices = market_summary.get("indices", [])
                
                html += f"""
                <div class="section">
                    <h2>Market Summary</h2>
                    <p>{market_summary.get("summary", "")}</p>
                    
                    <h3>Major Indices</h3>
                    <table>
                        <tr>
                            <th>Index</th>
                            <th>Value</th>
                            <th>Daily Change</th>
                            <th>Weekly Change</th>
                            <th>Monthly Change</th>
                        </tr>
                """
                
                for idx in indices:
                    name = idx.get("name", "")
                    value = idx.get("current_value", 0)
                    day_change = idx.get("day_change", 0)
                    week_change = idx.get("week_change", 0)
                    month_change = idx.get("month_change", 0)
                    
                    day_class = self._get_change_class(day_change)
                    week_class = self._get_change_class(week_change)
                    month_class = self._get_change_class(month_change)
                    
                    html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{value:.2f}</td>
                        <td class="{day_class}">{day_change:.2f}%</td>
                        <td class="{week_class}">{week_change:.2f}%</td>
                        <td class="{month_class}">{month_change:.2f}%</td>
                    </tr>
                    """
                
                html += """
                    </table>
                    
                    <h3>Market Breadth</h3>
                """
                
                breadth = market_summary.get("market_breadth", {})
                if breadth:
                    advances = breadth.get("advances", 0)
                    declines = breadth.get("declines", 0)
                    unchanged = breadth.get("unchanged", 0)
                    ratio = breadth.get("advance_decline_ratio", 0)
                    
                    html += f"""
                    <table>
                        <tr>
                            <th>Advances</th>
                            <td>{advances}</td>
                            <th>Declines</th>
                            <td>{declines}</td>
                        </tr>
                        <tr>
                            <th>Unchanged</th>
                            <td>{unchanged}</td>
                            <th>A/D Ratio</th>
                            <td>{ratio:.2f}</td>
                        </tr>
                    </table>
                    """
                
                html += """
                </div>
                """
            
            # 2. Index Performance Section
            if "index_performance" in sections:
                index_performance = sections["index_performance"]
                indices = index_performance.get("indices", [])
                
                html += f"""
                <div class="section">
                    <h2>Index Performance</h2>
                    <p>{index_performance.get("summary", "")}</p>
                    
                    <h3>Performance by Timeframe</h3>
                    <table>
                        <tr>
                            <th>Index</th>
                            <th>Daily</th>
                            <th>Weekly</th>
                            <th>Monthly</th>
                            <th>Quarterly</th>
                            <th>Yearly</th>
                        </tr>
                """
                
                for idx in indices:
                    name = idx.get("name", "")
                    returns = idx.get("returns", {})
                    
                    day = returns.get("day", 0)
                    week = returns.get("week", 0)
                    month = returns.get("month", 0)
                    quarter = returns.get("quarter", 0)
                    year = returns.get("year", 0)
                    
                    day_class = self._get_change_class(day)
                    week_class = self._get_change_class(week)
                    month_class = self._get_change_class(month)
                    quarter_class = self._get_change_class(quarter)
                    year_class = self._get_change_class(year)
                    
                    html += f"""
                    <tr>
                        <td>{name}</td>
                        <td class="{day_class}">{day:.2f}%</td>
                        <td class="{week_class}">{week:.2f}%</td>
                        <td class="{month_class}">{month:.2f}%</td>
                        <td class="{quarter_class}">{quarter:.2f}%</td>
                        <td class="{year_class}">{year:.2f}%</td>
                    </tr>
                    """
                
                html += """
                    </table>
                    
                    <h3>Moving Average Analysis</h3>
                    <table>
                        <tr>
                            <th>Index</th>
                            <th>50-day MA</th>
                            <th>200-day MA</th>
                            <th>Relation to 50-day</th>
                            <th>Relation to 200-day</th>
                        </tr>
                """
                
                for idx in indices:
                    name = idx.get("name", "")
                    ma_data = idx.get("moving_averages", {})
                    
                    ma_50 = ma_data.get("ma_50", 0)
                    ma_200 = ma_data.get("ma_200", 0)
                    vs_ma50 = ma_data.get("vs_ma50", 0)
                    vs_ma200 = ma_data.get("vs_ma200", 0)
                    
                    vs_ma50_class = self._get_change_class(vs_ma50)
                    vs_ma200_class = self._get_change_class(vs_ma200)
                    
                    html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{ma_50:.2f}</td>
                        <td>{ma_200:.2f}</td>
                        <td class="{vs_ma50_class}">{vs_ma50:.2f}%</td>
                        <td class="{vs_ma200_class}">{vs_ma200:.2f}%</td>
                    </tr>
                    """
                
                html += """
                    </table>
                """
                
                # Add charts
                charts = index_performance.get("charts", {})
                if charts:
                    html += """
                    <h3>Index Charts</h3>
                    <div class="charts-container">
                    """
                    
                    for idx_name, chart_data in charts.items():
                        html += f"""
                        <div class="chart">
                            <h4>{idx_name}</h4>
                            <img src="data:image/png;base64,{chart_data}" alt="{idx_name} Chart">
                        </div>
                        """
                    
                    html += """
                    </div>
                    """
                
                html += """
                </div>
                """
            
            # 3. Sector Performance Section
            if "sector_performance" in sections:
                sector_performance = sections["sector_performance"]
                sectors = sector_performance.get("sectors", [])
                
                html += f"""
                <div class="section">
                    <h2>Sector Performance</h2>
                    <p>{sector_performance.get("summary", "")}</p>
                    
                    <h3>Performance by Sector</h3>
                    <table>
                        <tr>
                            <th>Sector</th>
                            <th>Daily Change</th>
                            <th>Weekly Change</th>
                            <th>Monthly Change</th>
                        </tr>
                """
                
                for sector in sectors:
                    name = sector.get("name", "")
                    day_change = sector.get("day_change", 0)
                    week_change = sector.get("week_change", 0)
                    month_change = sector.get("month_change", 0)
                    
                    day_class = self._get_change_class(day_change)
                    week_class = self._get_change_class(week_change)
                    month_class = self._get_change_class(month_change)
                    
                    html += f"""
                    <tr>
                        <td>{name}</td>
                        <td class="{day_class}">{day_change:.2f}%</td>
                        <td class="{week_class}">{week_change:.2f}%</td>
                        <td class="{month_class}">{month_change:.2f}%</td>
                    </tr>
                    """
                
                html += """
                    </table>
                """
                
                # Add sector chart
                chart = sector_performance.get("chart", "")
                if chart:
                    html += f"""
                    <div class="chart">
                        <img src="data:image/png;base64,{chart}" alt="Sector Performance Chart">
                    </div>
                    """
                
                html += """
                </div>
                """
            
            # 4. Market Breadth Section
            if "market_breadth" in sections:
                market_breadth = sections["market_breadth"]
                
                html += f"""
                <div class="section">
                    <h2>Market Breadth</h2>
                    <p>{market_breadth.get("summary", "")}</p>
                    
                    <h3>Breadth Indicators</h3>
                    <table>
                """
                
                current_breadth = market_breadth.get("current_breadth", {})
                if current_breadth:
                    advances = current_breadth.get("advances", 0)
                    declines = current_breadth.get("declines", 0)
                    unchanged = current_breadth.get("unchanged", 0)
                    ratio = current_breadth.get("advance_decline_ratio", 0)
                    new_highs = current_breadth.get("new_highs", 0)
                    new_lows = current_breadth.get("new_lows", 0)
                    
                    html += f"""
                    <tr>
                        <th>Advances</th>
                        <td>{advances}</td>
                        <th>Declines</th>
                        <td>{declines}</td>
                    </tr>
                    <tr>
                        <th>Unchanged</th>
                        <td>{unchanged}</td>
                        <th>A/D Ratio</th>
                        <td>{ratio:.2f}</td>
                    </tr>
                    <tr>
                        <th>New 52-week Highs</th>
                        <td>{new_highs}</td>
                        <th>New 52-week Lows</th>
                        <td>{new_lows}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                """
                
                # Add breadth chart
                chart = market_breadth.get("chart", "")
                if chart:
                    html += f"""
                    <div class="chart">
                        <img src="data:image/png;base64,{chart}" alt="Market Breadth Chart">
                    </div>
                    """
                
                html += """
                </div>
                """
            
            # 5. Volatility Overview Section
            if "volatility_overview" in sections:
                volatility_overview = sections["volatility_overview"]
                
                html += f"""
                <div class="section">
                    <h2>Volatility Overview</h2>
                    <p>{volatility_overview.get("summary", "")}</p>
                    
                    <h3>Index Volatility</h3>
                    <table>
                        <tr>
                            <th>Index</th>
                            <th>Current Volatility</th>
                            <th>Historical Volatility</th>
                            <th>Volatility Ratio</th>
                            <th>State</th>
                        </tr>
                """
                
                index_volatility = volatility_overview.get("index_volatility", [])
                for item in index_volatility:
                    idx_name = item.get("index", "")
                    current = item.get("current_volatility", 0)
                    historical = item.get("historical_volatility", 0)
                    ratio = item.get("volatility_ratio", 0)
                    state = item.get("volatility_state", "stable")
                    
                    state_class = "neutral"
                    if state == "expanding":
                        state_class = "bearish"  # Higher volatility often associated with market stress
                    elif state == "contracting":
                        state_class = "bullish"  # Lower volatility often associated with bullish trends
                    
                    html += f"""
                    <tr>
                        <td>{idx_name}</td>
                        <td>{current:.2f}%</td>
                        <td>{historical:.2f}%</td>
                        <td>{ratio:.2f}</td>
                        <td class="{state_class}">{state.capitalize()}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                """
                
                # Add VIX data if available
                vix_data = volatility_overview.get("vix_data", {})
                if "current_value" in vix_data:
                    current_vix = vix_data.get("current_value", 0)
                    avg_vix = vix_data.get("average_value", 0)
                    percentile = vix_data.get("percentile", 50)
                    
                    html += f"""
                    <h3>Volatility Index (VIX)</h3>
                    <table>
                        <tr>
                            <th>Current Value</th>
                            <td>{current_vix:.2f}</td>
                            <th>Average Value</th>
                            <td>{avg_vix:.2f}</td>
                        </tr>
                        <tr>
                            <th>Percentile Rank</th>
                            <td>{percentile:.0f}%</td>
                            <th>Volatility Regime</th>
                            <td>{volatility_overview.get("volatility_regime", "normal").capitalize()}</td>
                        </tr>
                    </table>
                    """
                
                # Add volatility chart
                chart = volatility_overview.get("chart", "")
                if chart:
                    html += f"""
                    <div class="chart">
                        <img src="data:image/png;base64,{chart}" alt="Volatility Chart">
                    </div>
                    """
                
                html += """
                </div>
                """
            
            # 6. Correlation Overview Section
            if "correlation_overview" in sections:
                correlation_overview = sections["correlation_overview"]
                
                html += f"""
                <div class="section">
                    <h2>Correlation Overview</h2>
                    <p>{correlation_overview.get("summary", "")}</p>
                    
                    <h3>Correlation Matrix</h3>
                    <table>
                        <tr>
                            <th>Symbol</th>
                """
                
                # Get symbols from first row of matrix
                correlation_matrix = correlation_overview.get("correlation_matrix", [])
                if correlation_matrix:
                    first_row = correlation_matrix[0]
                    symbols = [key for key in first_row.keys() if key != "symbol"]
                    
                    # Add header row
                    for symbol in symbols:
                        html += f"""
                            <th>{symbol}</th>
                        """
                    
                    html += """
                        </tr>
                    """
                    
                    # Add data rows
                    for row in correlation_matrix:
                        row_symbol = row.get("symbol", "")
                        
                        html += f"""
                        <tr>
                            <th>{row_symbol}</th>
                        """
                        
                        for symbol in symbols:
                            corr_value = row.get(symbol, 0)
                            corr_class = self._get_correlation_class(corr_value)
                            
                            html += f"""
                            <td class="{corr_class}">{corr_value:.2f}</td>
                            """
                        
                        html += """
                        </tr>
                        """
                
                html += """
                    </table>
                    
                    <h3>Notable Correlation Pairs</h3>
                    <table>
                        <tr>
                            <th>Symbol 1</th>
                            <th>Symbol 2</th>
                            <th>Correlation</th>
                        </tr>
                """
                
                # High correlation pairs
                high_pairs = correlation_overview.get("high_correlation_pairs", [])
                for pair in high_pairs[:5]:  # Top 5
                    symbol1 = pair.get("symbol1", "")
                    symbol2 = pair.get("symbol2", "")
                    corr = pair.get("correlation", 0)
                    
                    html += f"""
                    <tr>
                        <td>{symbol1}</td>
                        <td>{symbol2}</td>
                        <td class="bullish">{corr:.2f}</td>
                    </tr>
                    """
                
                # Inverse correlation pairs
                inverse_pairs = correlation_overview.get("inverse_correlation_pairs", [])
                for pair in inverse_pairs[:5]:  # Top 5
                    symbol1 = pair.get("symbol1", "")
                    symbol2 = pair.get("symbol2", "")
                    corr = pair.get("correlation", 0)
                    
                    html += f"""
                    <tr>
                        <td>{symbol1}</td>
                        <td>{symbol2}</td>
                        <td class="bearish">{corr:.2f}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 7. Sentiment Analysis Section
            if "sentiment_analysis" in sections:
                sentiment_analysis = sections["sentiment_analysis"]
                
                html += f"""
                <div class="section">
                    <h2>Sentiment Analysis</h2>
                    <p>{sentiment_analysis.get("summary", "")}</p>
                    
                    <h3>Sentiment Breakdown</h3>
                    <table>
                """
                
                # News sentiment
                news = sentiment_analysis.get("news_sentiment", {})
                if news:
                    positive = news.get("positive_percent", 0)
                    negative = news.get("negative_percent", 0)
                    neutral = news.get("neutral_percent", 0)
                    
                    html += f"""
                    <tr>
                        <th colspan="2">News Sentiment</th>
                    </tr>
                    <tr>
                        <td>Positive</td>
                        <td class="bullish">{positive:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Negative</td>
                        <td class="bearish">{negative:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Neutral</td>
                        <td class="neutral">{neutral:.1f}%</td>
                    </tr>
                    """
                
                # Social sentiment
                social = sentiment_analysis.get("social_sentiment", {})
                if social:
                    bullish = social.get("bullish_percent", 0)
                    bearish = social.get("bearish_percent", 0)
                    neutral = social.get("neutral_percent", 0)
                    
                    html += f"""
                    <tr>
                        <th colspan="2">Social Media Sentiment</th>
                    </tr>
                    <tr>
                        <td>Bullish</td>
                        <td class="bullish">{bullish:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Bearish</td>
                        <td class="bearish">{bearish:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Neutral</td>
                        <td class="neutral">{neutral:.1f}%</td>
                    </tr>
                    """
                
                # Technical sentiment
                technical = sentiment_analysis.get("technical_sentiment", {})
                if technical:
                    bullish_indicators = technical.get("bullish_indicators", 0)
                    bearish_indicators = technical.get("bearish_indicators", 0)
                    total = bullish_indicators + bearish_indicators
                    
                    if total > 0:
                        bullish_pct = bullish_indicators / total * 100
                        bearish_pct = bearish_indicators / total * 100
                        
                        html += f"""
                        <tr>
                            <th colspan="2">Technical Sentiment</th>
                        </tr>
                        <tr>
                            <td>Bullish Indicators</td>
                            <td class="bullish">{bullish_pct:.1f}%</td>
                        </tr>
                        <tr>
                            <td>Bearish Indicators</td>
                            <td class="bearish">{bearish_pct:.1f}%</td>
                        </tr>
                        """
                
                html += """
                    </table>
                    
                    <h3>Overall Sentiment</h3>
                """
                
                overall = sentiment_analysis.get("overall_sentiment", "neutral")
                sentiment_class = "neutral"
                
                if "bullish" in overall:
                    sentiment_class = "bullish"
                elif "bearish" in overall:
                    sentiment_class = "bearish"
                
                html += f"""
                <p class="{sentiment_class}">{overall.replace('_', ' ').capitalize()}</p>
                """
                
                html += """
                </div>
                """
            
            # 8. Upcoming Events Section
            if "upcoming_events" in sections:
                upcoming_events = sections["upcoming_events"]
                
                html += f"""
                <div class="section">
                    <h2>Upcoming Events</h2>
                    <p>{upcoming_events.get("summary", "")}</p>
                    
                    <h3>Economic Events</h3>
                    <table>
                        <tr>
                            <th>Date</th>
                            <th>Event</th>
                            <th>Impact</th>
                            <th>Forecast</th>
                        </tr>
                """
                
                economic_events = upcoming_events.get("economic_events", [])
                for event in economic_events:
                    name = event.get("name", "")
                    date = event.get("date", "")
                    impact = event.get("impact", "medium")
                    forecast = event.get("forecast", "")
                    
                    if isinstance(date, datetime):
                        date_str = date.strftime("%Y-%m-%d %H:%M")
                    else:
                        date_str = str(date)
                    
                    impact_class = "neutral"
                    if impact == "high":
                        impact_class = "bearish"  # High impact events often cause volatility
                    
                    html += f"""
                    <tr>
                        <td>{date_str}</td>
                        <td>{name}</td>
                        <td class="{impact_class}">{impact.capitalize()}</td>
                        <td>{forecast}</td>
                    </tr>
                    """
                
                if not economic_events:
                    html += """
                    <tr>
                        <td colspan="4">No upcoming economic events</td>
                    </tr>
                    """
                
                html += """
                    </table>
                    
                    <h3>Earnings Reports</h3>
                    <table>
                        <tr>
                            <th>Date</th>
                            <th>Company</th>
                            <th>Expected EPS</th>
                            <th>Previous EPS</th>
                        </tr>
                """
                
                earnings_events = upcoming_events.get("earnings_events", [])
                for event in earnings_events:
                    company = event.get("company", "")
                    date = event.get("date", "")
                    expected_eps = event.get("expected_eps", "")
                    previous_eps = event.get("previous_eps", "")
                    
                    if isinstance(date, datetime):
                        date_str = date.strftime("%Y-%m-%d")
                    else:
                        date_str = str(date)
                    
                    html += f"""
                    <tr>
                        <td>{date_str}</td>
                        <td>{company}</td>
                        <td>{expected_eps}</td>
                        <td>{previous_eps}</td>
                    </tr>
                    """
                
                if not earnings_events:
                    html += """
                    <tr>
                        <td colspan="4">No upcoming earnings reports</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # Close HTML document
            html += """
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error formatting market report: {e}")
            return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"
    
    def _get_portfolio_data(self, portfolio_id: str = None) -> Dict[str, Any]:
        """
        Get portfolio data from database.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dictionary with portfolio data
        """
        try:
            query = {}
            if portfolio_id:
                query["portfolio_id"] = portfolio_id
            
            # Get the most recent portfolio snapshot
            portfolio = self.db.portfolio_snapshot_collection.find_one(
                query,
                sort=[("timestamp", -1)]
            )
            
            if not portfolio:
                # Try to build a portfolio from positions
                positions = list(self.db.positions_collection.find(query))
                
                if positions:
                    portfolio = {
                        "portfolio_id": portfolio_id,
                        "timestamp": datetime.now(),
                        "positions": positions
                    }
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio data: {e}")
            return {}
    
    def _generate_portfolio_summary(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate portfolio summary section.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            Dictionary with portfolio summary
        """
        try:
            positions = portfolio.get("positions", [])
            
            if not positions:
                return {
                    "status": "empty",
                    "message": "Portfolio has no positions"
                }
            
            # Calculate basic metrics
            total_value = sum(pos.get("current_value", 0) for pos in positions)
            total_cost = sum(pos.get("cost_basis", 0) for pos in positions)
            total_profit = total_value - total_cost
            
            if total_cost > 0:
                total_return = total_profit / total_cost * 100
            else:
                total_return = 0
            
            # Count positions by direction
            long_positions = [pos for pos in positions if pos.get("direction", "long") == "long"]
            short_positions = [pos for pos in positions if pos.get("direction", "long") == "short"]
            
            long_value = sum(pos.get("current_value", 0) for pos in long_positions)
            short_value = sum(pos.get("current_value", 0) for pos in short_positions)
            
            # Calculate sector allocation
            sector_allocation = {}
            for pos in positions:
                sector = pos.get("sector", "Unknown")
                value = pos.get("current_value", 0)
                
                if sector not in sector_allocation:
                    sector_allocation[sector] = 0
                
                sector_allocation[sector] += value
            
            # Convert to percentage
            sector_percentages = []
            if total_value > 0:
                for sector, value in sector_allocation.items():
                    percentage = value / total_value * 100
                    sector_percentages.append({
                        "sector": sector,
                        "value": value,
                        "percentage": percentage
                    })
            
            # Sort by percentage (descending)
            sector_percentages.sort(key=lambda x: x["percentage"], reverse=True)
            
            # Get recent performance
            day_change = sum(pos.get("day_change_amount", 0) for pos in positions)
            week_change = sum(pos.get("week_change_amount", 0) for pos in positions)
            month_change = sum(pos.get("month_change_amount", 0) for pos in positions)
            
            if total_value > 0:
                day_change_pct = day_change / total_value * 100
                week_change_pct = week_change / total_value * 100
                month_change_pct = month_change / total_value * 100
            else:
                day_change_pct = 0
                week_change_pct = 0
                month_change_pct = 0
            
            # Generate summary text
            summary = self._generate_portfolio_summary_text(
                total_value, total_profit, total_return,
                long_value, short_value, sector_percentages,
                day_change_pct, week_change_pct, month_change_pct
            )
            
            return {
                "total_value": total_value,
                "total_cost": total_cost,
                "total_profit": total_profit,
                "total_return": total_return,
                "position_count": len(positions),
                "long_position_count": len(long_positions),
                "short_position_count": len(short_positions),
                "long_value": long_value,
                "short_value": short_value,
                "sector_allocation": sector_percentages,
                "performance": {
                    "day_change": day_change,
                    "day_change_percent": day_change_pct,
                    "week_change": week_change,
                    "week_change_percent": week_change_pct,
                    "month_change": month_change,
                    "month_change_percent": month_change_pct
                },
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_portfolio_summary_text(self, total_value: float, total_profit: float,
                                      total_return: float, long_value: float, short_value: float,
                                      sector_allocation: List[Dict[str, Any]],
                                      day_change_pct: float, week_change_pct: float,
                                      month_change_pct: float) -> str:
        """
        Generate portfolio summary text.
        
        Args:
            total_value: Total portfolio value
            total_profit: Total profit
            total_return: Total return percentage
            long_value: Total long position value
            short_value: Total short position value
            sector_allocation: Sector allocation data
            day_change_pct: Daily change percentage
            week_change_pct: Weekly change percentage
            month_change_pct: Monthly change percentage
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Overall value and return
        summary_parts.append(f"The portfolio has a total value of ₹{total_value:,.2f}.")
        
        if total_profit > 0:
            summary_parts.append(f"Overall profit is ₹{total_profit:,.2f} ({total_return:.2f}%).")
        else:
            summary_parts.append(f"Overall loss is ₹{abs(total_profit):,.2f} ({total_return:.2f}%).")
        
        # Long/short exposure
        long_percentage = long_value / (long_value + short_value) * 100 if (long_value + short_value) > 0 else 0
        short_percentage = 100 - long_percentage
        
        if short_value > 0:
            summary_parts.append(f"The portfolio has {long_percentage:.1f}% long exposure and {short_percentage:.1f}% short exposure.")
        else:
            summary_parts.append("The portfolio consists entirely of long positions.")
        
        # Sector allocation
        if sector_allocation:
            top_sectors = sector_allocation[:3]
            top_sectors_str = ", ".join(f"{s['sector']} ({s['percentage']:.1f}%)" for s in top_sectors)
            summary_parts.append(f"Top sectors by allocation: {top_sectors_str}.")
        
        # Recent performance
        if day_change_pct > 0:
            summary_parts.append(f"The portfolio is up {day_change_pct:.2f}% today.")
        else:
            summary_parts.append(f"The portfolio is down {abs(day_change_pct):.2f}% today.")
        
        if month_change_pct > 0:
            summary_parts.append(f"Monthly performance is positive at {month_change_pct:.2f}%.")
        else:
            summary_parts.append(f"Monthly performance is negative at {abs(month_change_pct):.2f}%.")
        
        return " ".join(summary_parts)
    
    def _generate_performance_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate portfolio performance metrics section.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get performance history
            performance_history = self._get_portfolio_performance_history(portfolio.get("portfolio_id"))
            
            if not performance_history:
                return {
                    "status": "not_available",
                    "message": "Performance history not available"
                }
            
            # Calculate metrics
            current_value = performance_history[-1].get("value", 0)
            
            # Get performance for different timeframes
            daily = self._calculate_performance_for_timeframe(performance_history, 1)
            weekly = self._calculate_performance_for_timeframe(performance_history, 7)
            monthly = self._calculate_performance_for_timeframe(performance_history, 30)
            quarterly = self._calculate_performance_for_timeframe(performance_history, 90)
            yearly = self._calculate_performance_for_timeframe(performance_history, 365)
            
            # Calculate risk metrics
            returns = []
            for i in range(1, len(performance_history)):
                prev_value = performance_history[i-1].get("value", 0)
                curr_value = performance_history[i].get("value", 0)
                
                if prev_value > 0:
                    daily_return = (curr_value / prev_value - 1) * 100
                    returns.append(daily_return)
            
            if returns:
                volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
                avg_return = np.mean(returns) * 252  # Annualized return
                
                # Sharpe ratio (assuming risk-free rate of 4%)
                risk_free_rate = 4.0
                sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
                
                # Maximum drawdown
                max_drawdown = 0
                peak = performance_history[0].get("value", 0)
                
                for point in performance_history:
                    value = point.get("value", 0)
                    peak = max(peak, value)
                    drawdown = (peak - value) / peak * 100 if peak > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                volatility = 0
                avg_return = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            # Generate summary text
            summary = self._generate_performance_metrics_summary(
                daily, weekly, monthly, yearly,
                volatility, sharpe_ratio, max_drawdown
            )
            
            return {
                "current_value": current_value,
                "timeframe_performance": {
                    "daily": daily,
                    "weekly": weekly,
                    "monthly": monthly,
                    "quarterly": quarterly,
                    "yearly": yearly
                },
                "risk_metrics": {
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown
                },
                "performance_history": performance_history,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance metrics: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_portfolio_performance_history(self, portfolio_id: str = None) -> List[Dict[str, Any]]:
        """
        Get portfolio performance history.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            List of portfolio performance history points
        """
        try:
            query = {}
            if portfolio_id:
                query["portfolio_id"] = portfolio_id
            
            # Get portfolio history
            cursor = self.db.portfolio_history_collection.find(
                query
            ).sort("timestamp", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio performance history: {e}")
            return []
    
    def _calculate_performance_for_timeframe(self, history: List[Dict[str, Any]], 
                                          days: int) -> Dict[str, Any]:
        """
        Calculate performance for a specific timeframe.
        
        Args:
            history: Portfolio performance history
            days: Number of days for timeframe
            
        Returns:
            Dictionary with performance metrics
        """
        if not history or len(history) < 2:
            return {
                "change": 0,
                "change_percent": 0
            }
        
        current_value = history[-1].get("value", 0)
        current_date = history[-1].get("timestamp", datetime.now())
        
        # Find the starting point
        start_date = current_date - timedelta(days=days)
        start_value = None
        
        for point in reversed(history):
            point_date = point.get("timestamp", datetime.now())
            if point_date <= start_date:
                start_value = point.get("value", 0)
                break
        
        # If no starting point found, use the earliest available
        if start_value is None:
            start_value = history[0].get("value", 0)
        
        # Calculate change
        change = current_value - start_value
        change_percent = change / start_value * 100 if start_value > 0 else 0
        
        return {
            "change": change,
            "change_percent": change_percent
        }
    
    def _generate_performance_metrics_summary(self, daily: Dict[str, Any],
                                          weekly: Dict[str, Any],
                                          monthly: Dict[str, Any],
                                          yearly: Dict[str, Any],
                                          volatility: float,
                                          sharpe_ratio: float,
                                          max_drawdown: float) -> str:
        """
        Generate performance metrics summary text.
        
        Args:
            daily: Daily performance
            weekly: Weekly performance
            monthly: Monthly performance
            yearly: Yearly performance
            volatility: Portfolio volatility
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Recent performance
        daily_change = daily.get("change_percent", 0)
        if daily_change > 0:
            summary_parts.append(f"The portfolio is up {daily_change:.2f}% today.")
        else:
            summary_parts.append(f"The portfolio is down {abs(daily_change):.2f}% today.")
        
        # Monthly performance
        monthly_change = monthly.get("change_percent", 0)
        if monthly_change > 0:
            summary_parts.append(f"Monthly performance is positive at {monthly_change:.2f}%.")
        else:
            summary_parts.append(f"Monthly performance is negative at {abs(monthly_change):.2f}%.")
        
        # Yearly performance
        yearly_change = yearly.get("change_percent", 0)
        if yearly_change > 0:
            summary_parts.append(f"Yearly performance is positive at {yearly_change:.2f}%.")
        else:
            summary_parts.append(f"Yearly performance is negative at {abs(yearly_change):.2f}%.")
        
        # Risk metrics
        summary_parts.append(f"The portfolio has an annualized volatility of {volatility:.2f}%.")
        
        if sharpe_ratio > 1.0:
            summary_parts.append(f"Sharpe ratio of {sharpe_ratio:.2f} indicates good risk-adjusted returns.")
        elif sharpe_ratio > 0:
            summary_parts.append(f"Sharpe ratio of {sharpe_ratio:.2f} indicates moderate risk-adjusted returns.")
        else:
            summary_parts.append(f"Sharpe ratio of {sharpe_ratio:.2f} indicates poor risk-adjusted returns.")
        
        summary_parts.append(f"Maximum drawdown of {max_drawdown:.2f}% represents the largest decline from a peak.")
        
        return " ".join(summary_parts)
    
    def _generate_position_analysis(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate position analysis section.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            Dictionary with position analysis
        """
        try:
            positions = portfolio.get("positions", [])
            
            if not positions:
                return {
                    "status": "empty",
                    "message": "Portfolio has no positions"
                }
            
            # Sort positions by value (descending)
            positions.sort(key=lambda x: x.get("current_value", 0), reverse=True)
            
            # Calculate position metrics
            position_metrics = []
            
            for pos in positions:
                symbol = pos.get("symbol", "")
                entry_price = pos.get("entry_price", 0)
                current_price = pos.get("current_price", 0)
                quantity = pos.get("quantity", 0)
                direction = pos.get("direction", "long")
                
                # Calculate profit/loss
                if direction == "long":
                    profit_loss = (current_price - entry_price) * quantity
                    profit_loss_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
                else:
                    profit_loss = (entry_price - current_price) * quantity
                    profit_loss_pct = (entry_price / current_price - 1) * 100 if current_price > 0 else 0
                
                # Calculate contribution to portfolio
                current_value = pos.get("current_value", 0)
                portfolio_value = sum(p.get("current_value", 0) for p in positions)
                contribution = current_value / portfolio_value * 100 if portfolio_value > 0 else 0
                
                position_metrics.append({
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "quantity": quantity,
                    "direction": direction,
                    "current_value": current_value,
                    "profit_loss": profit_loss,
                    "profit_loss_percent": profit_loss_pct,
                    "contribution": contribution
                })
            
            # Generate summary text
            summary = self._generate_position_analysis_summary(position_metrics)
            
            return {
                "positions": position_metrics,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating position analysis: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_position_analysis_summary(self, positions: List[Dict[str, Any]]) -> str:
        """
        Generate position analysis summary text.
        
        Args:
            positions: Position metrics
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        if not positions:
            return "No positions to analyze."
        
        # Top positions
        top_positions = positions[:3]
        top_positions_str = ", ".join(f"{pos['symbol']} ({pos['contribution']:.1f}%)" for pos in top_positions)
        summary_parts.append(f"Top positions by value: {top_positions_str}.")
        
        # Winning positions
        winning_positions = [pos for pos in positions if pos.get("profit_loss", 0) > 0]
        winning_positions.sort(key=lambda x: x.get("profit_loss_percent", 0), reverse=True)
        
        if winning_positions:
            top_winners = winning_positions[:3]
            top_winners_str = ", ".join(f"{pos['symbol']} ({pos['profit_loss_percent']:.2f}%)" for pos in top_winners)
            summary_parts.append(f"Best performing positions: {top_winners_str}.")
        
        # Losing positions
        losing_positions = [pos for pos in positions if pos.get("profit_loss", 0) < 0]
        losing_positions.sort(key=lambda x: x.get("profit_loss_percent", 0))
        
        if losing_positions:
            top_losers = losing_positions[:3]
            top_losers_str = ", ".join(f"{pos['symbol']} ({pos['profit_loss_percent']:.2f}%)" for pos in top_losers)
            summary_parts.append(f"Worst performing positions: {top_losers_str}.")
        
        # Overall statistics
        total_positions = len(positions)
        winning_count = len(winning_positions)
        losing_count = len(losing_positions)
        
        win_ratio = winning_count / total_positions * 100 if total_positions > 0 else 0
        summary_parts.append(f"Win ratio: {win_ratio:.1f}% ({winning_count} winners, {losing_count} losers).")
        
        return " ".join(summary_parts)
    
    def _generate_risk_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate risk metrics section.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            Dictionary with risk metrics
        """
        try:
            positions = portfolio.get("positions", [])
            
            if not positions:
                return {
                    "status": "empty",
                    "message": "Portfolio has no positions"
                }
            
            # Calculate basic risk metrics
            total_value = sum(pos.get("current_value", 0) for pos in positions)
            
            # Concentration risk
            positions_sorted = sorted(positions, key=lambda x: x.get("current_value", 0), reverse=True)
            top_position = positions_sorted[0] if positions_sorted else {}
            top_position_pct = top_position.get("current_value", 0) / total_value * 100 if total_value > 0 else 0
            
            # Sector concentration
            sector_values = {}
            for pos in positions:
                sector = pos.get("sector", "Unknown")
                value = pos.get("current_value", 0)
                
                if sector not in sector_values:
                    sector_values[sector] = 0
                
                sector_values[sector] += value
            
            sector_percentages = []
            for sector, value in sector_values.items():
                percentage = value / total_value * 100 if total_value > 0 else 0
                sector_percentages.append({
                    "sector": sector,
                    "value": value,
                    "percentage": percentage
                })
            
            sector_percentages.sort(key=lambda x: x["percentage"], reverse=True)
            top_sector = sector_percentages[0] if sector_percentages else {}
            top_sector_pct = top_sector.get("percentage", 0)
            
            # Beta exposure
            # Get beta values for each position
            portfolio_beta = 0
            beta_available = False
            
            for pos in positions:
                symbol = pos.get("symbol", "")
                value = pos.get("current_value", 0)
                
                # Try to get beta from database
                stock_data = self.db.stock_metrics_collection.find_one(
                    {"symbol": symbol}
                )
                
                if stock_data and "beta" in stock_data:
                    beta = stock_data["beta"]
                    weighted_beta = beta * (value / total_value) if total_value > 0 else 0
                    portfolio_beta += weighted_beta
                    beta_available = True
            
            # VaR calculation (95% confidence, 1-day horizon)
            # Using simplified approach based on normal distribution
            portfolio_volatility = 0
            var_available = False
            
            # Get historical returns
            returns_data = {}
            
            for pos in positions:
                symbol = pos.get("symbol", "")
                
                # Get historical price data
                data = self._get_market_data(symbol, pos.get("exchange", "NSE"), days=252)
                
                if data and len(data) > 30:
                    df = pd.DataFrame(data).sort_values("timestamp")
                    returns = df["close"].pct_change().dropna().values
                    returns_data[symbol] = returns
            
            if returns_data:
                # Calculate portfolio volatility
                returns_matrix = []
                weights = []
                
                for pos in positions:
                    symbol = pos.get("symbol", "")
                    if symbol in returns_data:
                        returns_matrix.append(returns_data[symbol])
                        weights.append(pos.get("current_value", 0) / total_value if total_value > 0 else 0)
                
                if returns_matrix and weights:
                    # Convert to numpy arrays
                    returns_matrix = np.array(returns_matrix)
                    weights = np.array(weights)
                    
                    # Calculate covariance matrix
                    cov_matrix = np.cov(returns_matrix)
                    
                    # Calculate portfolio variance
                    portfolio_variance = weights.T @ cov_matrix @ weights
                    
                    # Calculate volatility
                    portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)  # Annualized
                    
                    # Calculate Value at Risk (95% confidence)
                    var_95 = norm.ppf(0.05) * portfolio_volatility / np.sqrt(252) * total_value
                    
                    var_available = True
            
            # Drawdown
            drawdown = 0
            
            # Get portfolio history
            history = self._get_portfolio_performance_history(portfolio.get("portfolio_id"))
            
            if history and len(history) > 1:
                peak = 0
                for point in history:
                    value = point.get("value", 0)
                    peak = max(peak, value)
                    current_drawdown = (peak - value) / peak * 100 if peak > 0 else 0
                    drawdown = max(drawdown, current_drawdown)
            
            # Generate summary text
            summary = self._generate_risk_metrics_summary(
                top_position_pct, top_sector_pct, portfolio_beta,
                portfolio_volatility, var_95 if var_available else None,
                drawdown
            )
            
            return {
                "concentration_risk": {
                    "top_position_percentage": top_position_pct,
                    "top_position": top_position.get("symbol", ""),
                    "top_sector_percentage": top_sector_pct,
                    "top_sector": top_sector.get("sector", "")
                },
                "market_risk": {
                    "portfolio_beta": portfolio_beta,
                    "beta_available": beta_available
                },
                "volatility_risk": {
                    "portfolio_volatility": portfolio_volatility,
                    "var_95": var_95 if var_available else None,
                    "var_available": var_available
                },
                "drawdown_risk": {
                    "max_drawdown": drawdown
                },
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk metrics: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_risk_metrics_summary(self, top_position_pct: float, top_sector_pct: float,
                                    portfolio_beta: float, portfolio_volatility: float,
                                    var_95: Optional[float], drawdown: float) -> str:
        """
        Generate risk metrics summary text.
        
        Args:
            top_position_pct: Top position percentage
            top_sector_pct: Top sector percentage
            portfolio_beta: Portfolio beta
            portfolio_volatility: Portfolio volatility
            var_95: Value at Risk (95% confidence)
            drawdown: Maximum drawdown
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Concentration risk
        if top_position_pct > 20:
            summary_parts.append(f"Position concentration is high with top position at {top_position_pct:.1f}% of the portfolio.")
        else:
            summary_parts.append(f"Position concentration is reasonable with top position at {top_position_pct:.1f}% of the portfolio.")
        
        if top_sector_pct > 50:
            summary_parts.append(f"Sector concentration is high with top sector at {top_sector_pct:.1f}% of the portfolio.")
        else:
            summary_parts.append(f"Sector concentration is reasonable with top sector at {top_sector_pct:.1f}% of the portfolio.")
        
        # Market risk
        if portfolio_beta > 1.2:
            summary_parts.append(f"The portfolio has high market sensitivity with a beta of {portfolio_beta:.2f}.")
        elif portfolio_beta < 0.8:
            summary_parts.append(f"The portfolio has low market sensitivity with a beta of {portfolio_beta:.2f}.")
        else:
            summary_parts.append(f"The portfolio has moderate market sensitivity with a beta of {portfolio_beta:.2f}.")
        
        # Volatility risk
        if portfolio_volatility > 25:
            summary_parts.append(f"Portfolio volatility is high at {portfolio_volatility:.2f}% annualized.")
        elif portfolio_volatility > 15:
            summary_parts.append(f"Portfolio volatility is moderate at {portfolio_volatility:.2f}% annualized.")
        else:
            summary_parts.append(f"Portfolio volatility is low at {portfolio_volatility:.2f}% annualized.")
        
        # VaR
        if var_95 is not None:
            var_pct = abs(var_95) / sum(p.get("current_value", 0) for p in self.positions) * 100 if self.positions else 0
            summary_parts.append(f"The 1-day 95% Value at Risk (VaR) is {abs(var_95):,.2f} ({var_pct:.2f}% of portfolio value).")
        
        # Drawdown
        if drawdown > 20:
            summary_parts.append(f"The maximum historical drawdown of {drawdown:.2f}% indicates significant downside risk.")
        elif drawdown > 10:
            summary_parts.append(f"The maximum historical drawdown of {drawdown:.2f}% indicates moderate downside risk.")
        else:
            summary_parts.append(f"The maximum historical drawdown of {drawdown:.2f}% indicates well-controlled downside risk.")
        
        return " ".join(summary_parts)
    
    def _generate_portfolio_correlation(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate portfolio correlation section.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            Dictionary with portfolio correlation
        """
        try:
            positions = portfolio.get("positions", [])
            
            if not positions or len(positions) < 2:
                return {
                    "status": "insufficient_positions",
                    "message": "At least two positions are required for correlation analysis"
                }
            
            # Extract symbols
            symbols = [pos.get("symbol", "") for pos in positions if pos.get("symbol")]
            
            if len(symbols) < 2:
                return {
                    "status": "insufficient_symbols",
                    "message": "At least two valid symbols are required for correlation analysis"
                }
            
            # Use correlation analyzer if available
            if self.correlation_analyzer:
                exchange = positions[0].get("exchange", "NSE")  # Assume same exchange for all positions
                
                corr_analysis = self.correlation_analyzer.analyze_correlation_matrix(symbols, exchange)
                
                if corr_analysis and "status" in corr_analysis and corr_analysis["status"] == "success":
                    # Extract relevant parts
                    correlation_matrix = corr_analysis.get("correlation_matrix", [])
                    high_pairs = corr_analysis.get("high_correlation_pairs", [])
                    inverse_pairs = corr_analysis.get("inverse_correlation_pairs", [])
                    avg_correlation = corr_analysis.get("average_correlation", 0)
                    
                    # Generate summary text
                    summary = self._generate_portfolio_correlation_summary(
                        correlation_matrix, high_pairs, inverse_pairs, avg_correlation
                    )
                    
                    return {
                        "correlation_matrix": correlation_matrix,
                        "high_correlation_pairs": high_pairs,
                        "inverse_correlation_pairs": inverse_pairs,
                        "average_correlation": avg_correlation,
                        "summary": summary
                    }
            
            # Fallback to simple correlation calculation
            return self._calculate_simple_portfolio_correlation(positions)
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio correlation: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_simple_portfolio_correlation(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate simple portfolio correlation.
        
        Args:
            positions: List of portfolio positions
            
        Returns:
            Dictionary with correlation data
        """
        try:
            # Extract symbols and exchange
            symbols = [pos.get("symbol", "") for pos in positions if pos.get("symbol")]
            exchange = positions[0].get("exchange", "NSE")  # Assume same exchange for all positions
            
            # Get price data for all symbols
            price_data = {}
            
            for symbol in symbols:
                data = self._get_market_data(symbol, exchange, days=252)
                
                if not data or len(data) < 30:
                    continue
                
                df = pd.DataFrame(data)
                
                # Extract closing prices with timestamps
                symbol_data = df[["timestamp", "close"]].sort_values("timestamp")
                price_data[symbol] = symbol_data
            
            if len(price_data) < 2:
                return {
                    "status": "insufficient_data",
                    "message": "Insufficient price data for correlation analysis"
                }
            
            # Create a combined dataframe with all closing prices
            combined_df = None
            
            for symbol, data in price_data.items():
                if combined_df is None:
                    combined_df = data.rename(columns={"close": symbol}).set_index("timestamp")
                else:
                    combined_df[symbol] = data.set_index("timestamp")["close"]
            
            # Fill missing values
            combined_df = combined_df.fillna(method="ffill").fillna(method="bfill")
            
            # Calculate returns
            returns_df = combined_df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Format correlation matrix for output
            formatted_matrix = []
            valid_symbols = list(correlation_matrix.columns)
            
            for symbol1 in valid_symbols:
                row = {"symbol": symbol1}
                for symbol2 in valid_symbols:
                    row[symbol2] = round(correlation_matrix.loc[symbol1, symbol2], 2)
                formatted_matrix.append(row)
            
            # Find highly correlated and inversely correlated pairs
            high_correlation_pairs = []
            inverse_correlation_pairs = []
            
            for i in range(len(valid_symbols)):
                for j in range(i+1, len(valid_symbols)):
                    symbol1 = valid_symbols[i]
                    symbol2 = valid_symbols[j]
                    corr = correlation_matrix.loc[symbol1, symbol2]
                    
                    if corr >= 0.7:
                        high_correlation_pairs.append({
                            "symbol1": symbol1,
                            "symbol2": symbol2,
                            "correlation": corr
                        })
                    elif corr <= -0.7:
                        inverse_correlation_pairs.append({
                            "symbol1": symbol1,
                            "symbol2": symbol2,
                            "correlation": corr
                        })
            
            # Calculate average correlation
            corr_values = []
            for i in range(len(valid_symbols)):
                for j in range(i+1, len(valid_symbols)):
                    corr_values.append(correlation_matrix.loc[valid_symbols[i], valid_symbols[j]])
            
            avg_correlation = sum(corr_values) / len(corr_values) if corr_values else 0
            
            # Generate summary text
            summary = self._generate_portfolio_correlation_summary(
                formatted_matrix, high_correlation_pairs, inverse_correlation_pairs, avg_correlation
            )
            
            return {
                "correlation_matrix": formatted_matrix,
                "high_correlation_pairs": high_correlation_pairs,
                "inverse_correlation_pairs": inverse_correlation_pairs,
                "average_correlation": avg_correlation,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating simple portfolio correlation: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_portfolio_correlation_summary(self, correlation_matrix: List[Dict[str, Any]],
                                             high_pairs: List[Dict[str, Any]],
                                             inverse_pairs: List[Dict[str, Any]],
                                             avg_correlation: float) -> str:
        """
        Generate portfolio correlation summary text.
        
        Args:
            correlation_matrix: Correlation matrix
            high_pairs: Highly correlated pairs
            inverse_pairs: Inversely correlated pairs
            avg_correlation: Average correlation
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Average correlation
        if avg_correlation > 0.7:
            summary_parts.append(f"The portfolio has high internal correlation ({avg_correlation:.2f}), indicating limited diversification benefit across holdings.")
        elif avg_correlation > 0.4:
            summary_parts.append(f"The portfolio has moderate internal correlation ({avg_correlation:.2f}).")
        else:
            summary_parts.append(f"The portfolio has low internal correlation ({avg_correlation:.2f}), indicating good diversification across holdings.")
        
        # High correlation pairs
        if high_pairs:
            top_pair = high_pairs[0]
            symbol1 = top_pair.get("symbol1", "")
            symbol2 = top_pair.get("symbol2", "")
            corr = top_pair.get("correlation", 0)
            
            summary_parts.append(f"The most highly correlated positions are {symbol1} and {symbol2} ({corr:.2f}), suggesting potential portfolio concentration.")
        
        # Inverse correlation pairs
        if inverse_pairs:
            top_pair = inverse_pairs[0]
            symbol1 = top_pair.get("symbol1", "")
            symbol2 = top_pair.get("symbol2", "")
            corr = top_pair.get("correlation", 0)
            
            summary_parts.append(f"The most inversely correlated positions are {symbol1} and {symbol2} ({corr:.2f}), providing natural hedging benefit.")
        
        # Portfolio implications
        if avg_correlation > 0.7 and not inverse_pairs:
            summary_parts.append("The high correlation across positions suggests opportunity for improved diversification by adding uncorrelated or negatively correlated assets.")
        elif inverse_pairs and avg_correlation < 0.5:
            summary_parts.append("The portfolio shows good diversification with balanced exposure across correlated and inversely correlated assets.")
        
        return " ".join(summary_parts)
    
    def _generate_sector_allocation(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate sector allocation section.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            Dictionary with sector allocation
        """
        try:
            positions = portfolio.get("positions", [])
            
            if not positions:
                return {
                    "status": "empty",
                    "message": "Portfolio has no positions"
                }
            
            # Calculate sector allocation
            sector_values = {}
            sector_positions = {}
            
            for pos in positions:
                sector = pos.get("sector", "Unknown")
                value = pos.get("current_value", 0)
                
                if sector not in sector_values:
                    sector_values[sector] = 0
                    sector_positions[sector] = []
                
                sector_values[sector] += value
                sector_positions[sector].append(pos)
            
            # Calculate total portfolio value
            total_value = sum(sector_values.values())
            
            # Calculate sector percentages
            sector_allocation = []
            
            for sector, value in sector_values.items():
                percentage = value / total_value * 100 if total_value > 0 else 0
                sector_allocation.append({
                    "sector": sector,
                    "value": value,
                    "percentage": percentage,
                    "positions": len(sector_positions[sector])
                })
            
            # Sort by percentage (descending)
            sector_allocation.sort(key=lambda x: x["percentage"], reverse=True)
            
            # Compare to benchmark allocation
            benchmark_allocation = self._get_benchmark_sector_allocation()
            
            sector_comparison = []
            
            if benchmark_allocation:
                for sector in sector_allocation:
                    sector_name = sector["sector"]
                    sector_pct = sector["percentage"]
                    
                    benchmark_pct = next((b["percentage"] for b in benchmark_allocation if b["sector"] == sector_name), 0)
                    
                    over_under = sector_pct - benchmark_pct
                    
                    sector_comparison.append({
                        "sector": sector_name,
                        "portfolio_percentage": sector_pct,
                        "benchmark_percentage": benchmark_pct,
                        "over_under_weight": over_under
                    })
            
            # Generate summary text
            summary = self._generate_sector_allocation_summary(
                sector_allocation, sector_comparison
            )
            
            return {
                "sector_allocation": sector_allocation,
                "sector_comparison": sector_comparison,
                "benchmark_allocation": benchmark_allocation,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating sector allocation: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_benchmark_sector_allocation(self) -> List[Dict[str, Any]]:
        """
        Get benchmark sector allocation.
        
        Returns:
            List of benchmark sector allocation
        """
        try:
            # Query the benchmark allocation collection
            benchmark = self.db.benchmark_allocation_collection.find_one(
                {},
                sort=[("timestamp", -1)]
            )
            
            if not benchmark or "sectors" not in benchmark:
                return []
            
            return benchmark["sectors"]
            
        except Exception as e:
            self.logger.error(f"Error getting benchmark sector allocation: {e}")
            return []
    
    def _generate_sector_allocation_summary(self, sector_allocation: List[Dict[str, Any]],
                                         sector_comparison: List[Dict[str, Any]]) -> str:
        """
        Generate sector allocation summary text.
        
        Args:
            sector_allocation: Sector allocation data
            sector_comparison: Sector comparison data
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Top sectors
        if sector_allocation:
            top_sectors = sector_allocation[:3]
            top_sectors_str = ", ".join(f"{s['sector']} ({s['percentage']:.1f}%)" for s in top_sectors)
            summary_parts.append(f"Top sectors by allocation: {top_sectors_str}.")
        
        # Over/underweight positions
        if sector_comparison:
            # Find most overweight and underweight sectors
            sector_comparison.sort(key=lambda x: x["over_under_weight"], reverse=True)
            
            if len(sector_comparison) > 0:
                most_overweight = sector_comparison[0]
                over_sector = most_overweight["sector"]
                over_weight = most_overweight["over_under_weight"]
                
                if over_weight > 5:
                    summary_parts.append(f"The portfolio is significantly overweight in {over_sector} ({over_weight:.1f}% above benchmark).")
                elif over_weight > 2:
                    summary_parts.append(f"The portfolio is moderately overweight in {over_sector} ({over_weight:.1f}% above benchmark).")
            
            if len(sector_comparison) > 1:
                most_underweight = sector_comparison[-1]
                under_sector = most_underweight["sector"]
                under_weight = most_underweight["over_under_weight"]
                
                if under_weight < -5:
                    summary_parts.append(f"The portfolio is significantly underweight in {under_sector} ({abs(under_weight):.1f}% below benchmark).")
                elif under_weight < -2:
                    summary_parts.append(f"The portfolio is moderately underweight in {under_sector} ({abs(under_weight):.1f}% below benchmark).")
        
        # Sector diversification
        if len(sector_allocation) >= 5:
            top_five_pct = sum(s["percentage"] for s in sector_allocation[:5])
            
            if top_five_pct > 80:
                summary_parts.append(f"Sector concentration is high with top 5 sectors comprising {top_five_pct:.1f}% of the portfolio.")
            elif top_five_pct < 60:
                summary_parts.append(f"Sector diversification is good with top 5 sectors comprising only {top_five_pct:.1f}% of the portfolio.")
            else:
                summary_parts.append(f"Sector diversification is moderate with top 5 sectors comprising {top_five_pct:.1f}% of the portfolio.")
        
        return " ".join(summary_parts)
    
    def _generate_optimization_suggestions(self, portfolio: Dict[str, Any], 
                                        sections: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate optimization suggestions section.
        
        Args:
            portfolio: Portfolio data
            sections: Report sections
            
        Returns:
            Dictionary with optimization suggestions
        """
        try:
            positions = portfolio.get("positions", [])
            
            if not positions:
                return {
                    "status": "empty",
                    "message": "Portfolio has no positions"
                }
            
            # Extract data from previous sections
            performance = sections.get("performance_metrics", {})
            risk = sections.get("risk_metrics", {})
            correlation = sections.get("correlation_matrix", {})
            sector_allocation = sections.get("sector_allocation", {})
            
            # Generate optimization suggestions
            suggestions = []
            
            # 1. Risk Management Suggestions
            risk_suggestions = self._generate_risk_suggestions(risk)
            suggestions.extend(risk_suggestions)
            
            # 2. Diversification Suggestions
            diversification_suggestions = self._generate_diversification_suggestions(correlation, sector_allocation)
            suggestions.extend(diversification_suggestions)
            
            # 3. Performance Enhancement Suggestions
            performance_suggestions = self._generate_performance_suggestions(performance, positions)
            suggestions.extend(performance_suggestions)
            
            # Generate summary text
            summary = self._generate_optimization_summary(suggestions)
            
            return {
                "suggestions": suggestions,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating optimization suggestions: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_risk_suggestions(self, risk: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate risk management suggestions.
        
        Args:
            risk: Risk metrics data
            
        Returns:
            List of risk management suggestions
        """
        suggestions = []
        
        # Concentration risk
        concentration = risk.get("concentration_risk", {})
        top_position_pct = concentration.get("top_position_percentage", 0)
        top_position = concentration.get("top_position", "")
        
        if top_position_pct > 20:
            suggestions.append({
                "category": "risk",
                "type": "position_concentration",
                "suggestion": f"Reduce exposure to {top_position} which currently accounts for {top_position_pct:.1f}% of the portfolio.",
                "reasoning": "High concentration in a single position increases idiosyncratic risk. Consider reducing this position to improve diversification."
            })
        
        top_sector_pct = concentration.get("top_sector_percentage", 0)
        top_sector = concentration.get("top_sector", "")
        
        if top_sector_pct > 50:
            suggestions.append({
                "category": "risk",
                "type": "sector_concentration",
                "suggestion": f"Reduce exposure to the {top_sector} sector which currently accounts for {top_sector_pct:.1f}% of the portfolio.",
                "reasoning": "High sector concentration exposes the portfolio to sector-specific risks. Consider increasing allocations to other sectors."
            })
        
        # Beta exposure
        market_risk = risk.get("market_risk", {})
        portfolio_beta = market_risk.get("portfolio_beta", 0)
        
        if portfolio_beta > 1.5:
            suggestions.append({
                "category": "risk",
                "type": "market_sensitivity",
                "suggestion": "Reduce overall portfolio beta by adding defensive stocks or increasing cash position.",
                "reasoning": f"Portfolio beta of {portfolio_beta:.2f} indicates high sensitivity to market movements, which could amplify losses in market downturns."
            })
        
        # Volatility
        volatility_risk = risk.get("volatility_risk", {})
        portfolio_volatility = volatility_risk.get("portfolio_volatility", 0)
        
        if portfolio_volatility > 25:
            suggestions.append({
                "category": "risk",
                "type": "volatility",
                "suggestion": "Reduce portfolio volatility by adding low-volatility assets or increasing positions in defensive sectors.",
                "reasoning": f"Annualized volatility of {portfolio_volatility:.2f}% is high and may lead to significant drawdowns."
            })
        
        return suggestions
    
    def _generate_diversification_suggestions(self, correlation: Dict[str, Any], 
                                           sector_allocation: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate diversification suggestions.
        
        Args:
            correlation: Correlation data
            sector_allocation: Sector allocation data
            
        Returns:
            List of diversification suggestions
        """
        suggestions = []
        
        # Correlation
        avg_correlation = correlation.get("average_correlation", 0)
        high_pairs = correlation.get("high_correlation_pairs", [])
        
        if avg_correlation > 0.7:
            suggestions.append({
                "category": "diversification",
                "type": "correlation",
                "suggestion": "Improve portfolio diversification by adding assets with low correlation to current holdings.",
                "reasoning": f"High average correlation of {avg_correlation:.2f} suggests limited diversification benefit across current holdings."
            })
        
        if high_pairs and len(high_pairs) > 2:
            symbol_pairs = ", ".join(f"{p['symbol1']}/{p['symbol2']}" for p in high_pairs[:2])
            suggestions.append({
                "category": "diversification",
                "type": "high_correlation_pairs",
                "suggestion": f"Consider reducing exposure to one asset in each highly correlated pair (e.g., {symbol_pairs}).",
                "reasoning": "Highly correlated assets provide redundant exposure and reduce effective diversification."
            })
        
        # Sector diversification
        allocation = sector_allocation.get("sector_allocation", [])
        comparison = sector_allocation.get("sector_comparison", [])
        
        if len(allocation) < 4:
            suggestions.append({
                "category": "diversification",
                "type": "sector_count",
                "suggestion": "Increase exposure to additional sectors to improve diversification.",
                "reasoning": f"Portfolio is currently concentrated in only {len(allocation)} sectors."
            })
        
        if comparison:
            # Find most underweight sectors
            underweight_sectors = [s for s in comparison if s.get("over_under_weight", 0) < -5]
            
            if underweight_sectors:
                sectors_str = ", ".join(s["sector"] for s in underweight_sectors[:2])
                suggestions.append({
                    "category": "diversification",
                    "type": "sector_underweight",
                    "suggestion": f"Consider adding exposure to underweight sectors such as {sectors_str}.",
                    "reasoning": "Increasing allocation to underweight sectors can improve diversification and potentially capture growth opportunities in those areas."
                })
        
        return suggestions
    
    def _generate_performance_suggestions(self, performance: Dict[str, Any], 
                                       positions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Generate performance enhancement suggestions.
        
        Args:
            performance: Performance metrics data
            positions: Portfolio positions
            
        Returns:
            List of performance enhancement suggestions
        """
        suggestions = []
        
        # Underperforming positions
        losing_positions = [pos for pos in positions if pos.get("profit_loss", 0) < 0]
        losing_positions.sort(key=lambda x: x.get("profit_loss_percent", 0))
        
        if losing_positions and len(losing_positions) > 2:
            worst_positions = losing_positions[:2]
            worst_str = ", ".join(f"{pos['symbol']} ({pos.get('profit_loss_percent', 0):.2f}%)" for pos in worst_positions)
            
            suggestions.append({
                "category": "performance",
                "type": "underperforming_positions",
                "suggestion": f"Consider exiting or reducing exposure to underperforming positions: {worst_str}.",
                "reasoning": "Continuing to hold significant underperforming positions may drag down overall portfolio performance."
            })
        
        # Performance metrics
        if "risk_metrics" in performance:
            risk_metrics = performance.get("risk_metrics", {})
            sharpe_ratio = risk_metrics.get("sharpe_ratio", 0)
            
            if sharpe_ratio < 0.5:
                suggestions.append({
                    "category": "performance",
                    "type": "risk_adjusted_return",
                    "suggestion": "Improve risk-adjusted returns by reducing exposure to high-volatility assets with inadequate returns.",
                    "reasoning": f"Low Sharpe ratio of {sharpe_ratio:.2f} indicates poor risk-adjusted performance."
                })
        
        # Asset allocation
        winners = [pos for pos in positions if pos.get("profit_loss", 0) > 0]
        if winners:
            winners.sort(key=lambda x: x.get("profit_loss_percent", 0), reverse=True)
            top_winners = winners[:2]
            top_str = ", ".join(f"{pos['symbol']}" for pos in top_winners)
            
            suggestions.append({
                "category": "performance",
                "type": "winning_positions",
                "suggestion": f"Consider increasing allocation to top-performing positions: {top_str}.",
                "reasoning": "Allocating more capital to strongest performers can help enhance overall returns, subject to concentration risk management."
            })
        
        return suggestions
    
    def _generate_optimization_summary(self, suggestions: List[Dict[str, str]]) -> str:
        """
        Generate optimization summary text.
        
        Args:
            suggestions: List of optimization suggestions
            
        Returns:
            Summary string
        """
        if not suggestions:
            return "No specific optimization recommendations at this time. Continue to monitor performance and market conditions."
        
        summary_parts = []
        
        # Group suggestions by category
        risk_suggestions = [s for s in suggestions if s.get("category") == "risk"]
        diversification_suggestions = [s for s in suggestions if s.get("category") == "diversification"]
        performance_suggestions = [s for s in suggestions if s.get("category") == "performance"]
        
        # Add summary for each category
        if risk_suggestions:
            summary_parts.append(f"Risk Management: {len(risk_suggestions)} recommendations to reduce portfolio risk, including addressing concentration and volatility concerns.")
        
        if diversification_suggestions:
            summary_parts.append(f"Diversification: {len(diversification_suggestions)} recommendations to improve portfolio diversification across assets and sectors.")
        
        if performance_suggestions:
            summary_parts.append(f"Performance Enhancement: {len(performance_suggestions)} recommendations to potentially improve portfolio returns.")
        
        # Overall optimization approach
        summary_parts.append("Consider implementing these recommendations gradually, prioritizing risk management and diversification before performance enhancement.")
        
        return " ".join(summary_parts)
    
    def _generate_market_charts(self, indices: List[str], exchange: str) -> Dict[str, str]:
        """
        Generate charts for market report.
        
        Args:
            indices: List of indices
            exchange: Stock exchange
            
        Returns:
            Dictionary with chart data
        """
        charts = {}
        
        try:
            # 1. Index Performance Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for idx in indices:
                data = self._get_market_data(idx, exchange, days=90)
                if not data or len(data) < 30:
                    continue
                
                df = pd.DataFrame(data).sort_values("timestamp")
                
                # Normalize to starting value
                start_value = df['close'].iloc[0]
                normalized = df['close'] / start_value * 100
                
                ax.plot(df['timestamp'], normalized, label=idx)
            
            ax.set_title('Normalized Index Performance (Last 90 Days)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Normalized Value (100 = Starting Value)')
            ax.legend()
            ax.grid(True)
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            charts['index_performance'] = img_str
            plt.close(fig)
            
            # 2. Market Breadth Chart
            try:
                # Get market breadth data
                breadth_data = self._get_historical_breadth_data(exchange)
                
                if breadth_data and len(breadth_data) > 10:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    dates = [data.get("timestamp") for data in breadth_data]
                    ad_ratio = [data.get("advance_decline_ratio", 1) for data in breadth_data]
                    
                    ax.plot(dates, ad_ratio, marker='o', linestyle='-', color='blue')
                    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
                    
                    ax.set_title('Advance-Decline Ratio')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('A/D Ratio')
                    ax.grid(True)
                    
                    # Convert figure to base64 string
                    buf = BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    charts['market_breadth'] = img_str
                    plt.close(fig)
            except Exception as e:
                self.logger.error(f"Error creating market breadth chart: {e}")
            
            # 3. Sector Performance Chart
            try:
                sectors = self._get_sector_performance_data(exchange)
                
                if sectors:
                    # Sort sectors by performance
                    sectors_sorted = sorted(sectors, key=lambda x: x.get("day_change", 0))
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    sector_names = [s["name"] for s in sectors_sorted]
                    returns = [s.get("day_change", 0) for s in sectors_sorted]
                    
                    # Set colors based on returns
                    colors = ['red' if ret < 0 else 'green' for ret in returns]
                    
                    # Create horizontal bar chart
                    bars = ax.barh(sector_names, returns, color=colors)
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
                                ha='left', va='center')
                    
                    ax.set_title('Sector Performance (%)')
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Convert figure to base64 string
                    buf = BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    charts['sector_performance'] = img_str
                    plt.close(fig)
            except Exception as e:
                self.logger.error(f"Error creating sector performance chart: {e}")
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error generating market charts: {e}")
            return {}
    
    def _generate_portfolio_charts(self, portfolio: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate charts for portfolio report.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            Dictionary with chart data
        """
        charts = {}
        
        try:
            positions = portfolio.get("positions", [])
            
            if not positions:
                return {}
            
            # 1. Portfolio Composition Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group small positions
            threshold = 3.0  # Positions less than 3% are grouped as "Other"
            
            position_values = {}
            portfolio_value = sum(pos.get("current_value", 0) for pos in positions)
            
            for pos in positions:
                symbol = pos.get("symbol", "")
                value = pos.get("current_value", 0)
                
                if portfolio_value > 0:
                    percentage = value / portfolio_value * 100
                else:
                    percentage = 0
                
                if percentage >= threshold:
                    position_values[symbol] = percentage
                else:
                    if "Other" not in position_values:
                        position_values["Other"] = 0
                    position_values["Other"] += percentage
            
            # Create pie chart
            labels = list(position_values.keys())
            sizes = list(position_values.values())
            
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title('Portfolio Composition (% of Value)')
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            charts['portfolio_composition'] = img_str
            plt.close(fig)
            
            # 2. Sector Allocation Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate sector allocation
            sector_values = {}
            
            for pos in positions:
                sector = pos.get("sector", "Unknown")
                value = pos.get("current_value", 0)
                
                if sector not in sector_values:
                    sector_values[sector] = 0
                
                sector_values[sector] += value
            
            # Calculate percentages
            sector_percentages = {}
            
            if portfolio_value > 0:
                for sector, value in sector_values.items():
                    sector_percentages[sector] = value / portfolio_value * 100
            
            # Create pie chart
            labels = list(sector_percentages.keys())
            sizes = list(sector_percentages.values())
            
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title('Sector Allocation (% of Value)')
            
            # Convert figure to base64 string
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            charts['sector_allocation'] = img_str
            plt.close(fig)
            
            # 3. Performance History Chart
            history = self._get_portfolio_performance_history(portfolio.get("portfolio_id"))
            
            if history and len(history) > 5:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                dates = [point.get("timestamp") for point in history]
                values = [point.get("value", 0) for point in history]
                
                ax.plot(dates, values, marker='o', linestyle='-', color='blue')
                
                ax.set_title('Portfolio Value History')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.grid(True)
                
                # Format y-axis as currency
                import matplotlib.ticker as mtick
                fmt = '₹{x:,.0f}'
                tick = mtick.StrMethodFormatter(fmt)
                ax.yaxis.set_major_formatter(tick)
                
                # Convert figure to base64 string
                buf = BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                charts['performance_history'] = img_str
                plt.close(fig)
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio charts: {e}")
            return {}
    
    def _format_portfolio_report(self, report: Dict[str, Any]) -> str:
        """
        Format portfolio report as HTML.
        
        Args:
            report: Report data
            
        Returns:
            HTML formatted report
        """
        try:
            # Get report sections
            sections = report.get("sections", {})
            portfolio_id = report.get("portfolio_id", "")
            generated_at = report.get("generated_at", datetime.now())
            
            # Start building HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Portfolio Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                    h1, h2, h3, h4 {{ color: #2c3e50; }}
                    .report-header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                    .chart {{ margin: 20px 0; text-align: center; }}
                    .chart img {{ max-width: 100%; height: auto; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .bullish {{ color: green; }}
                    .bearish {{ color: red; }}
                    .neutral {{ color: #888; }}
                    .summary-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1>Portfolio Analysis Report</h1>
                    <p>Generated on {generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """
            
            # 1. Portfolio Summary Section
            if "portfolio_summary" in sections:
                summary = sections["portfolio_summary"]
                
                html += f"""
                <div class="section">
                    <h2>Portfolio Summary</h2>
                    <p>{summary.get("summary", "")}</p>
                    
                    <h3>Portfolio Overview</h3>
                    <table>
                        <tr>
                            <th>Total Value</th>
                            <td>₹{summary.get("total_value", 0):,.2f}</td>
                            <th>Total Profit/Loss</th>
                            <td class="{self._get_change_class(summary.get("total_profit", 0))}">₹{summary.get("total_profit", 0):,.2f}</td>
                        </tr>
                        <tr>
                            <th>Total Return</th>
                            <td class="{self._get_change_class(summary.get("total_return", 0))}">{summary.get("total_return", 0):.2f}%</td>
                            <th>Number of Positions</th>
                            <td>{summary.get("position_count", 0)}</td>
                        </tr>
                        <tr>
                            <th>Long Positions</th>
                            <td>{summary.get("long_position_count", 0)}</td>
                            <th>Short Positions</th>
                            <td>{summary.get("short_position_count", 0)}</td>
                        </tr>
                    </table>
                    
                    <h3>Recent Performance</h3>
                    <table>
                        <tr>
                            <th>Daily Change</th>
                            <td class="{self._get_change_class(summary.get("performance", {}).get("day_change_percent", 0))}">{summary.get("performance", {}).get("day_change_percent", 0):.2f}%</td>
                            <th>Weekly Change</th>
                            <td class="{self._get_change_class(summary.get("performance", {}).get("week_change_percent", 0))}">{summary.get("performance", {}).get("week_change_percent", 0):.2f}%</td>
                        </tr>
                        <tr>
                            <th>Monthly Change</th>
                            <td class="{self._get_change_class(summary.get("performance", {}).get("month_change_percent", 0))}">{summary.get("performance", {}).get("month_change_percent", 0):.2f}%</td>
                            <th></th>
                            <td></td>
                        </tr>
                    </table>
                    
                    <div class="chart">
                        <img src="data:image/png;base64,{report.get('charts', {}).get('portfolio_composition', '')}" alt="Portfolio Composition Chart">
                    </div>
                </div>
                """
            
            # 2. Performance Metrics Section
            if "performance_metrics" in sections:
                performance = sections["performance_metrics"]
                
                html += f"""
                <div class="section">
                    <h2>Performance Metrics</h2>
                    <p>{performance.get("summary", "")}</p>
                    
                    <h3>Performance by Timeframe</h3>
                    <table>
                """
                
                timeframes = performance.get("timeframe_performance", {})
                
                if timeframes:
                    metrics = [
                        ("Daily", timeframes.get("daily", {})),
                        ("Weekly", timeframes.get("weekly", {})),
                        ("Monthly", timeframes.get("monthly", {})),
                        ("Quarterly", timeframes.get("quarterly", {})),
                        ("Yearly", timeframes.get("yearly", {}))
                    ]
                    
                    for name, metric in metrics:
                        change = metric.get("change", 0)
                        change_pct = metric.get("change_percent", 0)
                        
                        change_class = self._get_change_class(change)
                        
                        html += f"""
                        <tr>
                            <th>{name}</th>
                            <td class="{change_class}">₹{change:,.2f}</td>
                            <td class="{change_class}">{change_pct:.2f}%</td>
                        </tr>
                        """
                
                html += """
                    </table>
                    
                    <h3>Risk Metrics</h3>
                    <table>
                """
                
                risk_metrics = performance.get("risk_metrics", {})
                
                if risk_metrics:
                    volatility = risk_metrics.get("volatility", 0)
                    sharpe = risk_metrics.get("sharpe_ratio", 0)
                    drawdown = risk_metrics.get("max_drawdown", 0)
                    
                    html += f"""
                    <tr>
                        <th>Annualized Volatility</th>
                        <td>{volatility:.2f}%</td>
                        <th>Sharpe Ratio</th>
                        <td>{sharpe:.2f}</td>
                    </tr>
                    <tr>
                        <th>Maximum Drawdown</th>
                        <td>{drawdown:.2f}%</td>
                        <th></th>
                        <td></td>
                    </tr>
                    """
                
                html += """
                    </table>
                """
                
                # Add performance chart
                chart = report.get("charts", {}).get("performance_history", "")
                if chart:
                    html += f"""
                    <div class="chart">
                        <img src="data:image/png;base64,{chart}" alt="Performance History Chart">
                    </div>
                    """
                
                html += """
                </div>
                """
            
            # 3. Position Analysis Section
            if "position_analysis" in sections:
                position_analysis = sections["position_analysis"]
                
                html += f"""
                <div class="section">
                    <h2>Position Analysis</h2>
                    <p>{position_analysis.get("summary", "")}</p>
                    
                    <h3>Individual Positions</h3>
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th>Current Value</th>
                            <th>Profit/Loss</th>
                            <th>P/L %</th>
                            <th>Contribution</th>
                        </tr>
                """
                
                positions = position_analysis.get("positions", [])
                
                for pos in positions:
                    symbol = pos.get("symbol", "")
                    direction = pos.get("direction", "long")
                    value = pos.get("current_value", 0)
                    profit = pos.get("profit_loss", 0)
                    profit_pct = pos.get("profit_loss_percent", 0)
                    contribution = pos.get("contribution", 0)
                    
                    profit_class = self._get_change_class(profit)
                    
                    html += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td>{direction.capitalize()}</td>
                        <td>₹{value:,.2f}</td>
                        <td class="{profit_class}">₹{profit:,.2f}</td>
                        <td class="{profit_class}">{profit_pct:.2f}%</td>
                        <td>{contribution:.2f}%</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 4. Risk Metrics Section
            if "risk_metrics" in sections:
                risk_metrics = sections["risk_metrics"]
                
                html += f"""
                <div class="section">
                    <h2>Risk Assessment</h2>
                    <p>{risk_metrics.get("summary", "")}</p>
                    
                    <h3>Concentration Risk</h3>
                    <table>
                """
                
                concentration = risk_metrics.get("concentration_risk", {})
                
                if concentration:
                    top_pos = concentration.get("top_position", "")
                    top_pos_pct = concentration.get("top_position_percentage", 0)
                    top_sector = concentration.get("top_sector", "")
                    top_sector_pct = concentration.get("top_sector_percentage", 0)
                    
                    html += f"""
                    <tr>
                        <th>Top Position</th>
                        <td>{top_pos}</td>
                        <th>Percentage</th>
                        <td>{top_pos_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <th>Top Sector</th>
                        <td>{top_sector}</td>
                        <th>Percentage</th>
                        <td>{top_sector_pct:.2f}%</td>
                    </tr>
                    """
                
                html += """
                    </table>
                    
                    <h3>Market Risk</h3>
                    <table>
                """
                
                market_risk = risk_metrics.get("market_risk", {})
                
                if market_risk:
                    beta = market_risk.get("portfolio_beta", 0)
                    beta_available = market_risk.get("beta_available", False)
                    
                    if beta_available:
                        html += f"""
                        <tr>
                            <th>Portfolio Beta</th>
                            <td>{beta:.2f}</td>
                        </tr>
                        """
                    else:
                        html += """
                        <tr>
                            <th>Portfolio Beta</th>
                            <td>Not available</td>
                        </tr>
                        """
                
                html += """
                    </table>
                    
                    <h3>Volatility Risk</h3>
                    <table>
                """
                
                volatility_risk = risk_metrics.get("volatility_risk", {})
                
                if volatility_risk:
                    volatility = volatility_risk.get("portfolio_volatility", 0)
                    var_95 = volatility_risk.get("var_95")
                    var_available = volatility_risk.get("var_available", False)
                    
                    html += f"""
                    <tr>
                        <th>Portfolio Volatility (Annualized)</th>
                        <td>{volatility:.2f}%</td>
                    </tr>
                    """
                    
                    if var_available and var_95 is not None:
                        html += f"""
                        <tr>
                            <th>Value at Risk (95% 1-day)</th>
                            <td>₹{abs(var_95):,.2f}</td>
                        </tr>
                        """
                
                html += """
                    </table>
                    
                    <h3>Drawdown Risk</h3>
                    <table>
                """
                
                drawdown_risk = risk_metrics.get("drawdown_risk", {})
                
                if drawdown_risk:
                    max_drawdown = drawdown_risk.get("max_drawdown", 0)
                    
                    html += f"""
                    <tr>
                        <th>Maximum Historical Drawdown</th>
                        <td>{max_drawdown:.2f}%</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 5. Correlation Matrix Section
            if "correlation_matrix" in sections:
                correlation = sections["correlation_matrix"]
                
                html += f"""
                <div class="section">
                    <h2>Correlation Analysis</h2>
                    <p>{correlation.get("summary", "")}</p>
                    
                    <h3>Correlation Matrix</h3>
                    <table>
                        <tr>
                            <th>Symbol</th>
                """
                
                # Get symbols from first row of matrix
                correlation_matrix = correlation.get("correlation_matrix", [])
                if correlation_matrix:
                    first_row = correlation_matrix[0]
                    symbols = [key for key in first_row.keys() if key != "symbol"]
                    
                    # Add header row
                    for symbol in symbols:
                        html += f"""
                            <th>{symbol}</th>
                        """
                    
                    html += """
                        </tr>
                    """
                    
                    # Add data rows
                    for row in correlation_matrix:
                        row_symbol = row.get("symbol", "")
                        
                        html += f"""
                        <tr>
                            <th>{row_symbol}</th>
                        """
                        
                        for symbol in symbols:
                            corr_value = row.get(symbol, 0)
                            corr_class = self._get_correlation_class(corr_value)
                            
                            html += f"""
                            <td class="{corr_class}">{corr_value:.2f}</td>
                            """
                        
                        html += """
                        </tr>
                        """
                
                html += """
                    </table>
                    
                    <h3>Notable Correlation Pairs</h3>
                    <table>
                        <tr>
                            <th>Symbol 1</th>
                            <th>Symbol 2</th>
                            <th>Correlation</th>
                        </tr>
                """
                
                # High correlation pairs
                high_pairs = correlation.get("high_correlation_pairs", [])
                for pair in high_pairs[:5]:  # Top 5
                    symbol1 = pair.get("symbol1", "")
                    symbol2 = pair.get("symbol2", "")
                    corr = pair.get("correlation", 0)
                    
                    html += f"""
                    <tr>
                        <td>{symbol1}</td>
                        <td>{symbol2}</td>
                        <td class="bullish">{corr:.2f}</td>
                    </tr>
                    """
                
                # Inverse correlation pairs
                inverse_pairs = correlation.get("inverse_correlation_pairs", [])
                for pair in inverse_pairs[:5]:  # Top 5
                    symbol1 = pair.get("symbol1", "")
                    symbol2 = pair.get("symbol2", "")
                    corr = pair.get("correlation", 0)
                    
                    html += f"""
                    <tr>
                        <td>{symbol1}</td>
                        <td>{symbol2}</td>
                        <td class="bearish">{corr:.2f}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 6. Sector Allocation Section
            if "sector_allocation" in sections:
                sector_allocation = sections["sector_allocation"]
                
                html += f"""
                <div class="section">
                    <h2>Sector Allocation</h2>
                    <p>{sector_allocation.get("summary", "")}</p>
                    
                    <h3>Allocation by Sector</h3>
                    <table>
                        <tr>
                            <th>Sector</th>
                            <th>Value</th>
                            <th>Percentage</th>
                            <th>Positions</th>
                        </tr>
                """
                
                sectors = sector_allocation.get("sector_allocation", [])
                for sector in sectors:
                    sector_name = sector.get("sector", "")
                    value = sector.get("value", 0)
                    percentage = sector.get("percentage", 0)
                    positions = sector.get("positions", 0)
                    
                    html += f"""
                    <tr>
                        <td>{sector_name}</td>
                        <td>₹{value:,.2f}</td>
                        <td>{percentage:.2f}%</td>
                        <td>{positions}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                """
                
                # Add sector comparison if available
                comparison = sector_allocation.get("sector_comparison", [])
                if comparison:
                    html += """
                    <h3>Sector Comparison to Benchmark</h3>
                    <table>
                        <tr>
                            <th>Sector</th>
                            <th>Portfolio %</th>
                            <th>Benchmark %</th>
                            <th>Over/Under Weight</th>
                        </tr>
                """
                
                # Complete the sector comparison table
                for sector in comparison:
                    sector_name = sector.get("sector", "")
                    portfolio_pct = sector.get("portfolio_percentage", 0)
                    benchmark_pct = sector.get("benchmark_percentage", 0)
                    over_under = sector.get("over_under_weight", 0)
                    
                    over_under_class = "neutral"
                    if over_under > 5:
                        over_under_class = "bullish"
                    elif over_under < -5:
                        over_under_class = "bearish"
                    
                    html += f"""
                    <tr>
                        <td>{sector_name}</td>
                        <td>{portfolio_pct:.2f}%</td>
                        <td>{benchmark_pct:.2f}%</td>
                        <td class="{over_under_class}">{over_under:.2f}%</td>
                    </tr>
                    """
                
                html += """
                    </table>
                """
                
                # Add sector allocation chart
                chart = report.get("charts", {}).get("sector_allocation", "")
                if chart:
                    html += f"""
                    <div class="chart">
                        <img src="data:image/png;base64,{chart}" alt="Sector Allocation Chart">
                    </div>
                    """
                
                html += """
                </div>
                """
            
            # 7. Optimization Suggestions Section
            if "optimization_suggestions" in sections:
                optimization = sections["optimization_suggestions"]
                
                html += f"""
                <div class="section">
                    <h2>Optimization Suggestions</h2>
                    <p>{optimization.get("summary", "")}</p>
                    
                    <h3>Recommended Actions</h3>
                    <table>
                        <tr>
                            <th>Category</th>
                            <th>Suggestion</th>
                            <th>Reasoning</th>
                        </tr>
                """
                
                suggestions = optimization.get("suggestions", [])
                for suggestion in suggestions:
                    category = suggestion.get("category", "").capitalize()
                    text = suggestion.get("suggestion", "")
                    reasoning = suggestion.get("reasoning", "")
                    
                    html += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{text}</td>
                        <td>{reasoning}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # Close HTML document
            html += """
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error formatting portfolio report: {e}")
            return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"
    
    def _format_opportunity_report(self, report: Dict[str, Any]) -> str:
        """
        Format opportunity report as HTML.
        
        Args:
            report: Report data
            
        Returns:
            HTML formatted report
        """
        try:
            # Get report sections
            sections = report.get("sections", {})
            exchange = report.get("exchange", "")
            generated_at = report.get("generated_at", datetime.now())
            
            # Start building HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Opportunities Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                    h1, h2, h3, h4 {{ color: #2c3e50; }}
                    .report-header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .bullish {{ color: green; }}
                    .bearish {{ color: red; }}
                    .neutral {{ color: #888; }}
                    .summary-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1>Trading Opportunities Report: {exchange}</h1>
                    <p>Generated on {generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """
            
            # 1. Summary Section
            if "summary" in sections:
                summary = sections["summary"]
                
                html += f"""
                <div class="section">
                    <h2>Opportunities Summary</h2>
                    <p>{summary}</p>
                </div>
                """
            
            # 2. Breakout Opportunities Section
            if "breakout_opportunities" in sections:
                breakout_opportunities = sections["breakout_opportunities"]
                
                html += f"""
                <div class="section">
                    <h2>Breakout Opportunities</h2>
                    <p>{breakout_opportunities.get("summary", "")}</p>
                    
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th>Setup Type</th>
                            <th>Entry</th>
                            <th>Target</th>
                            <th>Stop</th>
                            <th>Risk/Reward</th>
                        </tr>
                """
                
                opportunities = breakout_opportunities.get("opportunities", [])
                for opp in opportunities:
                    symbol = opp.get("symbol", "")
                    direction = opp.get("direction", "")
                    setup = opp.get("setup", "")
                    entry = opp.get("entry", 0)
                    target = opp.get("target", 0)
                    stop = opp.get("stop", 0)
                    risk_reward = opp.get("risk_reward", 0)
                    
                    direction_class = "neutral"
                    if direction.lower() == "bullish" or direction.lower() == "long":
                        direction_class = "bullish"
                    elif direction.lower() == "bearish" or direction.lower() == "short":
                        direction_class = "bearish"
                    
                    html += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td class="{direction_class}">{direction}</td>
                        <td>{setup}</td>
                        <td>{entry:.2f}</td>
                        <td>{target:.2f}</td>
                        <td>{stop:.2f}</td>
                        <td>{risk_reward:.2f}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 3. Trend Following Opportunities Section
            if "trend_following_opportunities" in sections:
                trend_opportunities = sections["trend_following_opportunities"]
                
                html += f"""
                <div class="section">
                    <h2>Trend Following Opportunities</h2>
                    <p>{trend_opportunities.get("summary", "")}</p>
                    
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th>Setup Type</th>
                            <th>Entry</th>
                            <th>Target</th>
                            <th>Stop</th>
                            <th>Risk/Reward</th>
                        </tr>
                """
                
                opportunities = trend_opportunities.get("opportunities", [])
                for opp in opportunities:
                    symbol = opp.get("symbol", "")
                    direction = opp.get("direction", "")
                    setup = opp.get("setup", "")
                    entry = opp.get("entry", 0)
                    target = opp.get("target", 0)
                    stop = opp.get("stop", 0)
                    risk_reward = opp.get("risk_reward", 0)
                    
                    direction_class = "neutral"
                    if direction.lower() == "bullish" or direction.lower() == "long":
                        direction_class = "bullish"
                    elif direction.lower() == "bearish" or direction.lower() == "short":
                        direction_class = "bearish"
                    
                    html += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td class="{direction_class}">{direction}</td>
                        <td>{setup}</td>
                        <td>{entry:.2f}</td>
                        <td>{target:.2f}</td>
                        <td>{stop:.2f}</td>
                        <td>{risk_reward:.2f}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 4. Mean Reversion Opportunities Section
            if "mean_reversion_opportunities" in sections:
                mean_reversion_opportunities = sections["mean_reversion_opportunities"]
                
                html += f"""
                <div class="section">
                    <h2>Mean Reversion Opportunities</h2>
                    <p>{mean_reversion_opportunities.get("summary", "")}</p>
                    
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th>Setup Type</th>
                            <th>Entry</th>
                            <th>Target</th>
                            <th>Stop</th>
                            <th>Risk/Reward</th>
                        </tr>
                """
                
                opportunities = mean_reversion_opportunities.get("opportunities", [])
                for opp in opportunities:
                    symbol = opp.get("symbol", "")
                    direction = opp.get("direction", "")
                    setup = opp.get("setup", "")
                    entry = opp.get("entry", 0)
                    target = opp.get("target", 0)
                    stop = opp.get("stop", 0)
                    risk_reward = opp.get("risk_reward", 0)
                    
                    direction_class = "neutral"
                    if direction.lower() == "bullish" or direction.lower() == "long":
                        direction_class = "bullish"
                    elif direction.lower() == "bearish" or direction.lower() == "short":
                        direction_class = "bearish"
                    
                    html += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td class="{direction_class}">{direction}</td>
                        <td>{setup}</td>
                        <td>{entry:.2f}</td>
                        <td>{target:.2f}</td>
                        <td>{stop:.2f}</td>
                        <td>{risk_reward:.2f}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 5. Volatility Opportunities Section
            if "volatility_opportunities" in sections:
                volatility_opportunities = sections["volatility_opportunities"]
                
                html += f"""
                <div class="section">
                    <h2>Volatility Trading Opportunities</h2>
                    <p>{volatility_opportunities.get("summary", "")}</p>
                    
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Strategy</th>
                            <th>Current Volatility</th>
                            <th>Historical Volatility</th>
                            <th>Vol Ratio</th>
                            <th>Setup</th>
                        </tr>
                """
                
                opportunities = volatility_opportunities.get("opportunities", [])
                for opp in opportunities:
                    symbol = opp.get("symbol", "")
                    strategy = opp.get("strategy", "")
                    current_vol = opp.get("current_volatility", 0)
                    hist_vol = opp.get("historical_volatility", 0)
                    vol_ratio = opp.get("volatility_ratio", 0)
                    setup = opp.get("setup", "")
                    
                    html += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td>{strategy}</td>
                        <td>{current_vol:.2f}%</td>
                        <td>{hist_vol:.2f}%</td>
                        <td>{vol_ratio:.2f}</td>
                        <td>{setup}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 6. Pair Trading Opportunities Section
            if "pair_trading_opportunities" in sections:
                pair_opportunities = sections["pair_trading_opportunities"]
                
                html += f"""
                <div class="section">
                    <h2>Pair Trading Opportunities</h2>
                    <p>{pair_opportunities.get("summary", "")}</p>
                    
                    <table>
                        <tr>
                            <th>Symbol 1</th>
                            <th>Symbol 2</th>
                            <th>Correlation</th>
                            <th>Z-Score</th>
                            <th>Direction</th>
                            <th>Setup</th>
                        </tr>
                """
                
                opportunities = pair_opportunities.get("opportunities", [])
                for opp in opportunities:
                    symbol1 = opp.get("symbol1", "")
                    symbol2 = opp.get("symbol2", "")
                    correlation = opp.get("correlation", 0)
                    zscore = opp.get("zscore", 0)
                    direction = opp.get("direction", "")
                    setup = opp.get("setup", "")
                    
                    direction_class = "neutral"
                    if direction.lower() == "bullish" or direction.lower() == "long":
                        direction_class = "bullish"
                    elif direction.lower() == "bearish" or direction.lower() == "short":
                        direction_class = "bearish"
                    
                    html += f"""
                    <tr>
                        <td>{symbol1}</td>
                        <td>{symbol2}</td>
                        <td>{correlation:.2f}</td>
                        <td>{zscore:.2f}</td>
                        <td class="{direction_class}">{direction}</td>
                        <td>{setup}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # 7. Support/Resistance Opportunities Section
            if "support_resistance_opportunities" in sections:
                sr_opportunities = sections["support_resistance_opportunities"]
                
                html += f"""
                <div class="section">
                    <h2>Support/Resistance Opportunities</h2>
                    <p>{sr_opportunities.get("summary", "")}</p>
                    
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th>Level Type</th>
                            <th>Price Level</th>
                            <th>Current Price</th>
                            <th>Distance</th>
                            <th>Setup</th>
                        </tr>
                """
                
                opportunities = sr_opportunities.get("opportunities", [])
                for opp in opportunities:
                    symbol = opp.get("symbol", "")
                    direction = opp.get("direction", "")
                    level_type = opp.get("level_type", "")
                    price_level = opp.get("price_level", 0)
                    current_price = opp.get("current_price", 0)
                    distance = opp.get("distance", 0)
                    setup = opp.get("setup", "")
                    
                    direction_class = "neutral"
                    if direction.lower() == "bullish" or direction.lower() == "long":
                        direction_class = "bullish"
                    elif direction.lower() == "bearish" or direction.lower() == "short":
                        direction_class = "bearish"
                    
                    html += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td class="{direction_class}">{direction}</td>
                        <td>{level_type}</td>
                        <td>{price_level:.2f}</td>
                        <td>{current_price:.2f}</td>
                        <td>{distance:.2f}%</td>
                        <td>{setup}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
            
            # Close HTML document
            html += """
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error formatting opportunity report: {e}")
            return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"
    
    def _save_report(self, report: Dict[str, Any]) -> bool:
        """
        Save report to database.
        
        Args:
            report: Report data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a copy of the report to modify
            report_copy = report.copy()
            
            # Remove large content and charts for database storage
            if "content" in report_copy:
                del report_copy["content"]
            
            if "charts" in report_copy:
                del report_copy["charts"]
            
            # Add timestamp
            report_copy["timestamp"] = datetime.now()
            
            # Insert into database
            self.db.reports_collection.insert_one(report_copy)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
            return False
    
    def _get_sector_symbols(self, sector: str, exchange: str) -> List[str]:
        """
        Get symbols for a specific sector.
        
        Args:
            sector: Sector name
            exchange: Stock exchange
            
        Returns:
            List of symbols
        """
        try:
            # Query the portfolio collection
            cursor = self.db.portfolio_collection.find(
                {
                    "sector": sector,
                    "exchange": exchange,
                    "status": "active"
                }
            )
            
            symbols = [doc["symbol"] for doc in cursor]
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error getting sector symbols: {e}")
            return []