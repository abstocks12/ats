"""
Fundamental Analysis System

This module provides comprehensive fundamental analysis capabilities for the automated trading system.
It analyzes financial data, valuation metrics, growth trends, and sector comparisons.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import math
import statistics

class FundamentalAnalyzer:
    """
    Provides fundamental analysis capabilities for trading decisions.
    Analyzes financial statements, ratios, growth metrics, and valuation models.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the fundamental analyzer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Get query optimizer if available
        self.query_optimizer = getattr(self.db, 'get_query_optimizer', lambda: None)()
        
        # Define analysis parameters
        self.analysis_params = {
            # Valuation thresholds
            "pe_low_threshold": 10.0,     # PE ratio below which is considered low
            "pe_high_threshold": 30.0,    # PE ratio above which is considered high
            "pb_low_threshold": 1.0,      # PB ratio below which is considered low
            "pb_high_threshold": 4.0,     # PB ratio above which is considered high
            "ev_ebitda_low": 6.0,         # EV/EBITDA below which is considered low
            "ev_ebitda_high": 15.0,       # EV/EBITDA above which is considered high
            
            # Profitability thresholds
            "good_roe": 15.0,             # ROE above which is considered good
            "great_roe": 25.0,            # ROE above which is considered excellent
            "good_roic": 10.0,            # ROIC above which is considered good
            "great_roic": 20.0,           # ROIC above which is considered excellent
            
            # Growth thresholds
            "strong_revenue_growth": 15.0, # Revenue growth above which is considered strong
            "strong_profit_growth": 20.0,  # Profit growth above which is considered strong
            
            # Financial health
            "safe_debt_equity": 1.0,      # Debt-to-equity ratio below which is considered safe
            "concerning_debt_equity": 2.0, # Debt-to-equity ratio above which is concerning
            "good_current_ratio": 1.5,    # Current ratio above which is considered good
            "good_interest_coverage": 5.0, # Interest coverage ratio above which is considered good
            
            # Dividend metrics
            "high_dividend_yield": 4.0,   # Dividend yield above which is considered high
            "good_payout_ratio": 50.0,    # Payout ratio below which is considered sustainable
            
            # Quality metrics
            "good_gross_margin": 30.0,    # Gross margin above which is considered good
            "good_net_margin": 10.0,      # Net margin above which is considered good
            
            # Analysis periods
            "short_term_quarters": 4,     # Number of quarters for short-term analysis
            "medium_term_quarters": 8,    # Number of quarters for medium-term analysis
            "long_term_quarters": 12      # Number of quarters for long-term analysis
        }
        
        # Define sector average benchmarks (for demonstration - in production, these would be dynamically calculated)
        self.sector_benchmarks = {
            "IT": {
                "pe_ratio": 25.0,
                "pb_ratio": 6.0,
                "roe": 22.0,
                "net_margin": 15.0,
                "revenue_growth": 12.0
            },
            "Banking": {
                "pe_ratio": 15.0,
                "pb_ratio": 2.0,
                "roe": 15.0,
                "net_margin": 18.0,
                "revenue_growth": 8.0
            },
            "Automotive": {
                "pe_ratio": 18.0,
                "pb_ratio": 3.0,
                "roe": 14.0,
                "net_margin": 8.0,
                "revenue_growth": 7.0
            },
            "FMCG": {
                "pe_ratio": 35.0,
                "pb_ratio": 8.0,
                "roe": 25.0,
                "net_margin": 12.0,
                "revenue_growth": 9.0
            },
            "Pharma": {
                "pe_ratio": 22.0,
                "pb_ratio": 4.0,
                "roe": 17.0,
                "net_margin": 14.0,
                "revenue_growth": 10.0
            },
            "Energy": {
                "pe_ratio": 12.0,
                "pb_ratio": 1.5,
                "roe": 12.0,
                "net_margin": 9.0,
                "revenue_growth": 5.0
            },
            "Metals": {
                "pe_ratio": 10.0,
                "pb_ratio": 1.2,
                "roe": 10.0,
                "net_margin": 7.0,
                "revenue_growth": 4.0
            },
            "default": {  # Default values if sector is unknown
                "pe_ratio": 20.0,
                "pb_ratio": 3.0,
                "roe": 15.0,
                "net_margin": 10.0,
                "revenue_growth": 8.0
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
    
    def analyze(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            self.logger.info(f"Analyzing fundamentals for {symbol} ({exchange})")
            
            # Get company info
            company_info = self._get_company_info(symbol, exchange)
            if not company_info:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "error",
                    "error": "Company information not found"
                }
            
            # Get financial data
            financial_data = self._get_financial_data(symbol, exchange)
            if not financial_data:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "error",
                    "error": "Financial data not found"
                }
            
            # Get quarterly results
            quarterly_results = self._get_quarterly_results(symbol, exchange)
            if not quarterly_results:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "error",
                    "error": "Quarterly results not found"
                }
            
            # Get peer comparison data
            peer_comparison = self._get_peer_comparison(symbol, exchange, company_info.get("sector"))
            
            # Calculate key financial ratios
            ratios = self._calculate_financial_ratios(financial_data, quarterly_results)
            
            # Analyze growth trends
            growth_trends = self._analyze_growth_trends(quarterly_results)
            
            # Perform valuation analysis
            valuation = self._analyze_valuation(financial_data, ratios, growth_trends, company_info.get("sector"))
            
            # Analyze financial health
            financial_health = self._analyze_financial_health(financial_data, ratios)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(financial_data, quarterly_results, ratios)
            
            # Compare with sector benchmarks
            sector_comparison = self._compare_with_sector(ratios, growth_trends, company_info.get("sector"))
            
            # Generate investment thesis
            investment_thesis = self._generate_investment_thesis(
                company_info, ratios, growth_trends, valuation, 
                financial_health, quality_metrics, sector_comparison
            )
            
            # Generate SWOT analysis
            swot_analysis = self._generate_swot_analysis(
                company_info, ratios, growth_trends, valuation,
                financial_health, quality_metrics, sector_comparison
            )
            
            # Calculate fundamental score
            fundamental_score = self._calculate_fundamental_score(
                ratios, growth_trends, valuation, financial_health, 
                quality_metrics, sector_comparison
            )
            
            # Assemble the analysis result
            result = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now(),
                "status": "success",
                "company_info": company_info,
                "financial_summary": {
                    "latest_quarter": quarterly_results[0] if quarterly_results else None,
                    "key_ratios": ratios,
                    "growth_trends": growth_trends,
                    "valuation": valuation,
                    "financial_health": financial_health,
                    "quality_metrics": quality_metrics
                },
                "sector_comparison": sector_comparison,
                "peer_comparison": peer_comparison,
                "investment_thesis": investment_thesis,
                "swot_analysis": swot_analysis,
                "fundamental_score": fundamental_score,
                "recommendation": self._generate_recommendation(fundamental_score)
            }
            
            # Save analysis result to database
            self._save_analysis(symbol, exchange, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing fundamentals for {symbol}: {e}")
            return {
                "symbol": symbol,
                "exchange": exchange,
                "status": "error",
                "error": str(e)
            }
    
    def _get_company_info(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get company information.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with company information
        """
        try:
            # Query the database for company information
            company_info = self.db.company_info_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            if not company_info:
                self.logger.warning(f"Company info not found for {symbol} ({exchange})")
                return None
            
            # Extract relevant info
            info = {
                "name": company_info.get("name"),
                "sector": company_info.get("sector"),
                "industry": company_info.get("industry"),
                "market_cap": company_info.get("market_cap"),
                "description": company_info.get("description"),
                "website": company_info.get("website"),
                "listing_date": company_info.get("listing_date"),
                "ceo": company_info.get("ceo"),
                "headquarters": company_info.get("headquarters")
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error retrieving company info for {symbol}: {e}")
            return None
    
    def _get_financial_data(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get financial data for the company.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with financial data
        """
        try:
            # Query the database for the latest financial data
            financial_data = self.db.financial_data_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            }, sort=[("report_date", -1)])
            
            if not financial_data:
                self.logger.warning(f"Financial data not found for {symbol} ({exchange})")
                return None
            
            # Extract relevant data
            data = {
                "report_date": financial_data.get("report_date"),
                "report_type": financial_data.get("report_type"),
                "period": financial_data.get("period"),
                "sales": financial_data.get("sales"),
                "expenses": financial_data.get("expenses"),
                "operating_profit": financial_data.get("operating_profit"),
                "net_profit": financial_data.get("net_profit"),
                "eps": financial_data.get("eps"),
                "balance_sheet": financial_data.get("balance_sheet", {}),
                "cash_flow": financial_data.get("cash_flow", {}),
                "key_ratios": financial_data.get("key_ratios", {})
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving financial data for {symbol}: {e}")
            return None
    
    def _get_quarterly_results(self, symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """
        Get quarterly results for the company.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            List of quarterly results sorted by date (newest first)
        """
        try:
            # Query the database for quarterly results
            cursor = self.db.quarterly_results_collection.find({
                "symbol": symbol,
                "exchange": exchange
            }).sort("report_date", -1).limit(12)  # Get last 12 quarters
            
            quarterly_results = list(cursor)
            
            if not quarterly_results:
                self.logger.warning(f"Quarterly results not found for {symbol} ({exchange})")
                return None
            
            return quarterly_results
            
        except Exception as e:
            self.logger.error(f"Error retrieving quarterly results for {symbol}: {e}")
            return None
    
    def _get_peer_comparison(self, symbol: str, exchange: str, sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Get peer comparison data.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            sector: Company sector
            
        Returns:
            Dictionary with peer comparison data
        """
        try:
            if not sector:
                return {
                    "peers": [],
                    "metrics": {}
                }
            
            # Query the database for peer companies in the same sector
            cursor = self.db.company_info_collection.find({
                "sector": sector,
                "exchange": exchange,
                "symbol": {"$ne": symbol}  # Exclude the current company
            }).limit(5)
            
            peers = list(cursor)
            
            if not peers:
                return {
                    "peers": [],
                    "metrics": {}
                }
            
            # Get peer symbols
            peer_symbols = [peer["symbol"] for peer in peers]
            
            # Get latest financial data for peers
            peer_data = {}
            
            for peer_symbol in peer_symbols:
                # Get latest financial ratios
                financial_data = self.db.financial_data_collection.find_one({
                    "symbol": peer_symbol,
                    "exchange": exchange
                }, sort=[("report_date", -1)])
                
                if financial_data and "key_ratios" in financial_data:
                    peer_data[peer_symbol] = {
                        "name": next((peer["name"] for peer in peers if peer["symbol"] == peer_symbol), peer_symbol),
                        "market_cap": next((peer["market_cap"] for peer in peers if peer["symbol"] == peer_symbol), 0),
                        "pe_ratio": financial_data["key_ratios"].get("pe_ratio", 0),
                        "pb_ratio": financial_data["key_ratios"].get("pb_ratio", 0),
                        "roe": financial_data["key_ratios"].get("roe", 0),
                        "net_margin": financial_data["key_ratios"].get("npm_percent", 0),
                        "debt_equity": financial_data["key_ratios"].get("debt_to_equity", 0)
                    }
            
            # Calculate average metrics
            metrics = {
                "avg_pe_ratio": np.mean([peer["pe_ratio"] for peer in peer_data.values() if peer["pe_ratio"] > 0]),
                "avg_pb_ratio": np.mean([peer["pb_ratio"] for peer in peer_data.values() if peer["pb_ratio"] > 0]),
                "avg_roe": np.mean([peer["roe"] for peer in peer_data.values() if peer["roe"] > 0]),
                "avg_net_margin": np.mean([peer["net_margin"] for peer in peer_data.values() if peer["net_margin"] > 0]),
                "avg_debt_equity": np.mean([peer["debt_equity"] for peer in peer_data.values() if peer["debt_equity"] > 0])
            }
            
            return {
                "peers": list(peer_data.values()),
                "metrics": metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving peer comparison data for {symbol}: {e}")
            return {
                "peers": [],
                "metrics": {}
            }
    
    def _calculate_financial_ratios(self, financial_data: Dict[str, Any], 
                                  quarterly_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate key financial ratios.
        
        Args:
            financial_data: Financial data dictionary
            quarterly_results: List of quarterly results
            
        Returns:
            Dictionary with financial ratios
        """
        ratios = {}
        
        # Use existing ratios if available
        if financial_data and "key_ratios" in financial_data:
            existing_ratios = financial_data["key_ratios"]
            
            # Copy existing ratios
            ratios = {
                "pe_ratio": existing_ratios.get("pe_ratio", 0),
                "pb_ratio": existing_ratios.get("pb_ratio", 0),
                "ev_ebitda": existing_ratios.get("ev_ebitda", 0),
                "roe": existing_ratios.get("roe", 0),
                "roce": existing_ratios.get("roce", 0),
                "roic": existing_ratios.get("roic", 0),
                "debt_equity": existing_ratios.get("debt_to_equity", 0),
                "current_ratio": existing_ratios.get("current_ratio", 0),
                "interest_coverage": existing_ratios.get("interest_coverage", 0),
                "gross_margin": existing_ratios.get("gpm_percent", 0),
                "operating_margin": existing_ratios.get("opm_percent", 0),
                "net_margin": existing_ratios.get("npm_percent", 0),
                "dividend_yield": existing_ratios.get("dividend_yield", 0),
                "payout_ratio": existing_ratios.get("payout_ratio", 0)
            }
        
        # Calculate TTM (Trailing Twelve Months) metrics
        if quarterly_results and len(quarterly_results) >= 4:
            # Get last 4 quarters
            last_4_quarters = quarterly_results[:4]
            
            # TTM Revenue
            ttm_revenue = sum(q.get("sales", 0) for q in last_4_quarters)
            
            # TTM Net Profit
            ttm_net_profit = sum(q.get("net_profit", 0) for q in last_4_quarters)
            
            # TTM EPS
            ttm_eps = sum(q.get("eps", 0) for q in last_4_quarters)
            
            # Add TTM metrics
            ratios["ttm_revenue"] = ttm_revenue
            ratios["ttm_net_profit"] = ttm_net_profit
            ratios["ttm_eps"] = ttm_eps
            
            # Calculate TTM margins if not already available
            if ttm_revenue > 0:
                ratios["ttm_net_margin"] = (ttm_net_profit / ttm_revenue) * 100
        
        return ratios
    
    def _analyze_growth_trends(self, quarterly_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze growth trends from quarterly results.
        
        Args:
            quarterly_results: List of quarterly results
            
        Returns:
            Dictionary with growth trends
        """
        if not quarterly_results or len(quarterly_results) < 4:
            return {
                "revenue_growth": {},
                "profit_growth": {},
                "margin_trends": {}
            }
        
        # Get YoY growth for each quarter
        quarters = []
        
        for i in range(len(quarterly_results) - 4):
            current_quarter = quarterly_results[i]
            previous_year_quarter = quarterly_results[i + 4]
            
            quarter_data = {
                "period": current_quarter.get("period"),
                "revenue_yoy": calculate_growth(
                    current_quarter.get("sales", 0), 
                    previous_year_quarter.get("sales", 0)
                ),
                "profit_yoy": calculate_growth(
                    current_quarter.get("net_profit", 0), 
                    previous_year_quarter.get("net_profit", 0)
                ),
                "eps_yoy": calculate_growth(
                    current_quarter.get("eps", 0), 
                    previous_year_quarter.get("eps", 0)
                )
            }
            
            quarters.append(quarter_data)
        
        # Calculate average growth rates
        if quarters:
            avg_revenue_growth = np.mean([q["revenue_yoy"] for q in quarters if q["revenue_yoy"] is not None])
            avg_profit_growth = np.mean([q["profit_yoy"] for q in quarters if q["profit_yoy"] is not None])
            avg_eps_growth = np.mean([q["eps_yoy"] for q in quarters if q["eps_yoy"] is not None])
        else:
            avg_revenue_growth = None
            avg_profit_growth = None
            avg_eps_growth = None
        
        # Analyze margin trends
        margins = []
        
        for q in quarterly_results:
            if "sales" in q and q["sales"] > 0:
                margin_data = {
                    "period": q.get("period"),
                    "gross_margin": (q.get("gross_profit", 0) / q["sales"]) * 100 if "gross_profit" in q else None,
                    "operating_margin": (q.get("operating_profit", 0) / q["sales"]) * 100 if "operating_profit" in q else None,
                    "net_margin": (q.get("net_profit", 0) / q["sales"]) * 100 if "net_profit" in q else None
                }
                
                margins.append(margin_data)
        
        # Determine margin trend (improving, stable, or declining)
        margin_trend = "stable"
        
        if margins and len(margins) >= 4:
            recent_margins = [m["net_margin"] for m in margins[:4] if m["net_margin"] is not None]
            if recent_margins:
                margin_slope = np.polyfit(range(len(recent_margins)), recent_margins, 1)[0]
                
                if margin_slope > 0.5:  # More than 0.5 percentage point increase per quarter
                    margin_trend = "improving"
                elif margin_slope < -0.5:  # More than 0.5 percentage point decrease per quarter
                    margin_trend = "declining"
        
        return {
            "revenue_growth": {
                "quarters": [{"period": q["period"], "growth": q["revenue_yoy"]} for q in quarters],
                "avg_growth": avg_revenue_growth,
                "trend": determine_trend([q["revenue_yoy"] for q in quarters if q["revenue_yoy"] is not None])
            },
            "profit_growth": {
                "quarters": [{"period": q["period"], "growth": q["profit_yoy"]} for q in quarters],
                "avg_growth": avg_profit_growth,
                "trend": determine_trend([q["profit_yoy"] for q in quarters if q["profit_yoy"] is not None])
            },
            "eps_growth": {
                "quarters": [{"period": q["period"], "growth": q["eps_yoy"]} for q in quarters],
                "avg_growth": avg_eps_growth,
                "trend": determine_trend([q["eps_yoy"] for q in quarters if q["eps_yoy"] is not None])
            },
            "margin_trends": {
                "quarters": margins,
                "trend": margin_trend
            }
        }
    
    def _analyze_valuation(self, financial_data: Dict[str, Any], ratios: Dict[str, Any], 
                         growth_trends: Dict[str, Any], sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze valuation metrics.
        
        Args:
            financial_data: Financial data dictionary
            ratios: Financial ratios dictionary
            growth_trends: Growth trends dictionary
            sector: Company sector
            
        Returns:
            Dictionary with valuation analysis
        """
        # Get sector benchmarks
        sector_data = self.sector_benchmarks.get(sector, self.sector_benchmarks["default"]) if sector else self.sector_benchmarks["default"]
        
        # Extract PE ratio
        pe_ratio = ratios.get("pe_ratio", 0)
        
        # Extract growth rates
        avg_profit_growth = growth_trends.get("profit_growth", {}).get("avg_growth")
        
        # Calculate PEG ratio if possible
        peg_ratio = None
        if pe_ratio and avg_profit_growth and pe_ratio > 0 and avg_profit_growth > 0:
            peg_ratio = pe_ratio / avg_profit_growth
        
        # Determine if the stock is undervalued, fairly valued, or overvalued
        valuation_status = "fairly_valued"
        
        if pe_ratio:
            if pe_ratio < self.analysis_params["pe_low_threshold"] and pe_ratio < sector_data["pe_ratio"] * 0.7:
                valuation_status = "undervalued"
            elif pe_ratio > self.analysis_params["pe_high_threshold"] and pe_ratio > sector_data["pe_ratio"] * 1.3:
                valuation_status = "overvalued"
            
            # Refine based on PEG ratio if available
            if peg_ratio:
                if peg_ratio < 1.0:
                    valuation_status = "undervalued" if valuation_status != "overvalued" else "fairly_valued"
                elif peg_ratio > 2.0:
                    valuation_status = "overvalued" if valuation_status != "undervalued" else "fairly_valued"
        
        # Determine relative valuation to sector
        sector_relative = "in_line"
        if pe_ratio and sector_data["pe_ratio"]:
            pe_to_sector = pe_ratio / sector_data["pe_ratio"]
            
            if pe_to_sector < 0.8:
                sector_relative = "discount"
            elif pe_to_sector > 1.2:
                sector_relative = "premium"
        
        # Calculate DCF implied value (simplified)
        dcf_value = None
        if "ttm_net_profit" in ratios and avg_profit_growth:
            # Simple DCF model assumptions
            current_profit = ratios["ttm_net_profit"]
            growth_rate = min(avg_profit_growth / 100, 0.25)  # Cap growth at 25%
            terminal_growth = 0.03  # 3% terminal growth
            discount_rate = 0.10  # 10% discount rate
            
            # 5-year projection
            projected_cash_flows = []
            for year in range(1, 6):
                profit = current_profit * ((1 + growth_rate) ** year)
                projected_cash_flows.append(profit)
            
            # Terminal value
            terminal_value = projected_cash_flows[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
            
            # Discount cash flows
            dcf_value = 0
            for i, cf in enumerate(projected_cash_flows):
                dcf_value += cf / ((1 + discount_rate) ** (i + 1))
            
            # Add discounted terminal value
            dcf_value += terminal_value / ((1 + discount_rate) ** 5)
        
        return {
            "pe_ratio": pe_ratio,
            "sector_pe": sector_data["pe_ratio"],
            "pb_ratio": ratios.get("pb_ratio", 0),
            "sector_pb": sector_data["pb_ratio"],
            "ev_ebitda": ratios.get("ev_ebitda", 0),
            "peg_ratio": peg_ratio,
            "status": valuation_status,
            "sector_relative": sector_relative,
            "dcf_value": dcf_value,
            "valuation_notes": self._generate_valuation_notes(pe_ratio, ratios.get("pb_ratio", 0), 
                                                            ratios.get("ev_ebitda", 0), peg_ratio, 
                                                            avg_profit_growth, sector_data)
        }
    
    def _generate_valuation_notes(self, pe_ratio: float, pb_ratio: float, ev_ebitda: float, 
                               peg_ratio: Optional[float], growth_rate: Optional[float], 
                               sector_data: Dict[str, float]) -> str:
        """
        Generate valuation analysis notes.
        
        Args:
            pe_ratio: PE ratio
            pb_ratio: PB ratio
            ev_ebitda: EV/EBITDA ratio
            peg_ratio: PEG ratio
            growth_rate: Growth rate
            sector_data: Sector benchmark data
            
        Returns:
            Valuation notes string
        """
        notes = []
        
        # PE ratio analysis
        if pe_ratio:
            if pe_ratio < self.analysis_params["pe_low_threshold"]:
                notes.append(f"P/E ratio of {pe_ratio:.2f} is below the threshold of {self.analysis_params['pe_low_threshold']}, suggesting potential undervaluation.")
            elif pe_ratio > self.analysis_params["pe_high_threshold"]:
                notes.append(f"P/E ratio of {pe_ratio:.2f} is above the threshold of {self.analysis_params['pe_high_threshold']}, indicating potential overvaluation.")
            
            # Compare to sector
            if sector_data["pe_ratio"]:
                pe_diff = ((pe_ratio / sector_data["pe_ratio"]) - 1) * 100
                if pe_diff < -20:
                    notes.append(f"P/E ratio is {abs(pe_diff):.1f}% below the sector average, trading at a significant discount.")
                elif pe_diff > 20:
                    notes.append(f"P/E ratio is {pe_diff:.1f}% above the sector average, trading at a premium.")
        
        # PB ratio analysis
        # PB ratio analysis
        if pb_ratio:
            if pb_ratio < self.analysis_params["pb_low_threshold"]:
                notes.append(f"P/B ratio of {pb_ratio:.2f} is below the threshold of {self.analysis_params['pb_low_threshold']}, which may indicate undervaluation relative to book value.")
            elif pb_ratio > self.analysis_params["pb_high_threshold"]:
                notes.append(f"P/B ratio of {pb_ratio:.2f} is above the threshold of {self.analysis_params['pb_high_threshold']}, suggesting premium valuation relative to book value.")
            
            # Compare to sector
            if sector_data["pb_ratio"]:
                pb_diff = ((pb_ratio / sector_data["pb_ratio"]) - 1) * 100
                if pb_diff < -20:
                    notes.append(f"P/B ratio is {abs(pb_diff):.1f}% below the sector average.")
                elif pb_diff > 20:
                    notes.append(f"P/B ratio is {pb_diff:.1f}% above the sector average.")
        
        # EV/EBITDA analysis
        if ev_ebitda:
            if ev_ebitda < self.analysis_params["ev_ebitda_low"]:
                notes.append(f"EV/EBITDA of {ev_ebitda:.2f} is below the threshold of {self.analysis_params['ev_ebitda_low']}, suggesting potential undervaluation.")
            elif ev_ebitda > self.analysis_params["ev_ebitda_high"]:
                notes.append(f"EV/EBITDA of {ev_ebitda:.2f} is above the threshold of {self.analysis_params['ev_ebitda_high']}, indicating potential overvaluation.")
        
        # PEG ratio analysis
        if peg_ratio:
            if peg_ratio < 1.0:
                notes.append(f"PEG ratio of {peg_ratio:.2f} is below 1.0, suggesting the stock may be undervalued relative to its growth rate.")
            elif peg_ratio > 2.0:
                notes.append(f"PEG ratio of {peg_ratio:.2f} is above 2.0, indicating the stock may be overvalued relative to its growth rate.")
            elif 1.0 <= peg_ratio <= 2.0:
                notes.append(f"PEG ratio of {peg_ratio:.2f} is between 1.0 and 2.0, suggesting fair valuation relative to growth.")
        
        # Growth rate context
        if growth_rate:
            if growth_rate > sector_data["revenue_growth"]:
                notes.append(f"Growth rate of {growth_rate:.2f}% is above the sector average of {sector_data['revenue_growth']}%, which may justify premium valuation multiples.")
            else:
                notes.append(f"Growth rate of {growth_rate:.2f}% is below or in line with the sector average of {sector_data['revenue_growth']}%.")
        
        # Return combined notes
        return " ".join(notes)
    
    def _analyze_financial_health(self, financial_data: Dict[str, Any], ratios: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze financial health metrics.
        
        Args:
            financial_data: Financial data dictionary
            ratios: Financial ratios dictionary
            
        Returns:
            Dictionary with financial health analysis
        """
        # Extract balance sheet data
        balance_sheet = financial_data.get("balance_sheet", {})
        
        # Extract cash flow data
        cash_flow = financial_data.get("cash_flow", {})
        
        # Get key metrics
        debt_equity = ratios.get("debt_equity", 0)
        current_ratio = ratios.get("current_ratio", 0)
        interest_coverage = ratios.get("interest_coverage", 0)
        
        # Determine debt level
        debt_level = "moderate"
        if debt_equity < self.analysis_params["safe_debt_equity"]:
            debt_level = "low"
        elif debt_equity > self.analysis_params["concerning_debt_equity"]:
            debt_level = "high"
        
        # Determine liquidity status
        liquidity_status = "adequate"
        if current_ratio < 1.0:
            liquidity_status = "poor"
        elif current_ratio > self.analysis_params["good_current_ratio"]:
            liquidity_status = "strong"
        
        # Determine interest coverage status
        interest_status = "adequate"
        if interest_coverage < 2.0:
            interest_status = "poor"
        elif interest_coverage > self.analysis_params["good_interest_coverage"]:
            interest_status = "strong"
        
        # Overall financial health
        if debt_level == "low" and liquidity_status == "strong":
            overall_health = "excellent"
        elif debt_level == "high" or liquidity_status == "poor":
            overall_health = "concerning"
        else:
            overall_health = "good"
        
        # Calculate free cash flow if available
        free_cash_flow = None
        if "operating_cash_flow" in cash_flow and "capital_expenditure" in cash_flow:
            free_cash_flow = cash_flow["operating_cash_flow"] - cash_flow["capital_expenditure"]
        
        # FCF to net income ratio (cash quality)
        fcf_to_net_income = None
        if free_cash_flow is not None and financial_data.get("net_profit", 0) > 0:
            fcf_to_net_income = free_cash_flow / financial_data["net_profit"]
        
        return {
            "debt_equity": debt_equity,
            "current_ratio": current_ratio,
            "interest_coverage": interest_coverage,
            "free_cash_flow": free_cash_flow,
            "fcf_to_net_income": fcf_to_net_income,
            "debt_level": debt_level,
            "liquidity_status": liquidity_status,
            "interest_status": interest_status,
            "overall_health": overall_health,
            "health_notes": self._generate_financial_health_notes(debt_equity, current_ratio, 
                                                                interest_coverage, 
                                                                fcf_to_net_income)
        }
    
    def _generate_financial_health_notes(self, debt_equity: float, current_ratio: float, 
                                      interest_coverage: float, fcf_to_net_income: Optional[float]) -> str:
        """
        Generate financial health analysis notes.
        
        Args:
            debt_equity: Debt-to-equity ratio
            current_ratio: Current ratio
            interest_coverage: Interest coverage ratio
            fcf_to_net_income: Free cash flow to net income ratio
            
        Returns:
            Financial health notes string
        """
        notes = []
        
        # Debt analysis
        if debt_equity < self.analysis_params["safe_debt_equity"]:
            notes.append(f"Low debt-to-equity ratio of {debt_equity:.2f} indicates conservative financial leverage.")
        elif debt_equity > self.analysis_params["concerning_debt_equity"]:
            notes.append(f"High debt-to-equity ratio of {debt_equity:.2f} suggests significant financial leverage, which may increase financial risk.")
        else:
            notes.append(f"Moderate debt-to-equity ratio of {debt_equity:.2f} indicates balanced financial leverage.")
        
        # Liquidity analysis
        if current_ratio < 1.0:
            notes.append(f"Current ratio of {current_ratio:.2f} is below 1.0, indicating potential short-term liquidity challenges.")
        elif current_ratio > self.analysis_params["good_current_ratio"]:
            notes.append(f"Strong current ratio of {current_ratio:.2f} suggests robust short-term liquidity position.")
        else:
            notes.append(f"Adequate current ratio of {current_ratio:.2f} indicates reasonable short-term liquidity.")
        
        # Interest coverage analysis
        if interest_coverage < 2.0:
            notes.append(f"Low interest coverage ratio of {interest_coverage:.2f} may indicate challenges in meeting interest obligations.")
        elif interest_coverage > self.analysis_params["good_interest_coverage"]:
            notes.append(f"Strong interest coverage ratio of {interest_coverage:.2f} demonstrates substantial capacity to service debt.")
        else:
            notes.append(f"Adequate interest coverage ratio of {interest_coverage:.2f} indicates reasonable ability to service debt.")
        
        # Cash flow quality
        if fcf_to_net_income is not None:
            if fcf_to_net_income > 1.2:
                notes.append(f"Free cash flow to net income ratio of {fcf_to_net_income:.2f} indicates high-quality earnings with strong cash conversion.")
            elif fcf_to_net_income < 0.8:
                notes.append(f"Free cash flow to net income ratio of {fcf_to_net_income:.2f} suggests earnings quality concerns with weak cash conversion.")
            else:
                notes.append(f"Free cash flow to net income ratio of {fcf_to_net_income:.2f} indicates reasonable cash conversion from earnings.")
        
        # Return combined notes
        return " ".join(notes)
    
    def _calculate_quality_metrics(self, financial_data: Dict[str, Any], 
                                 quarterly_results: List[Dict[str, Any]],
                                 ratios: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate business quality metrics.
        
        Args:
            financial_data: Financial data dictionary
            quarterly_results: List of quarterly results
            ratios: Financial ratios dictionary
            
        Returns:
            Dictionary with quality metrics
        """
        # Extract margin data
        gross_margin = ratios.get("gross_margin", 0)
        operating_margin = ratios.get("operating_margin", 0)
        net_margin = ratios.get("net_margin", 0)
        
        # Extract return metrics
        roe = ratios.get("roe", 0)
        roce = ratios.get("roce", 0)
        roic = ratios.get("roic", 0)
        
        # Determine margin quality
        margin_quality = "average"
        if gross_margin > self.analysis_params["good_gross_margin"] and net_margin > self.analysis_params["good_net_margin"]:
            margin_quality = "high"
        elif gross_margin < self.analysis_params["good_gross_margin"] * 0.7 or net_margin < self.analysis_params["good_net_margin"] * 0.7:
            margin_quality = "low"
        
        # Determine return on capital quality
        capital_efficiency = "average"
        if roe > self.analysis_params["great_roe"] and roic > self.analysis_params["great_roic"]:
            capital_efficiency = "excellent"
        elif roe > self.analysis_params["good_roe"] and roic > self.analysis_params["good_roic"]:
            capital_efficiency = "good"
        elif roe < self.analysis_params["good_roe"] * 0.7 or roic < self.analysis_params["good_roic"] * 0.7:
            capital_efficiency = "poor"
        
        # Calculate earnings stability
        earnings_stability = "moderate"
        if quarterly_results and len(quarterly_results) >= 8:
            # Calculate coefficient of variation of net profit
            profits = [q.get("net_profit", 0) for q in quarterly_results if q.get("net_profit") is not None]
            if profits and np.mean(profits) > 0:
                cv = np.std(profits) / np.mean(profits)
                
                if cv < 0.2:
                    earnings_stability = "high"
                elif cv > 0.5:
                    earnings_stability = "low"
        
        # Determine overall business quality
        if margin_quality == "high" and capital_efficiency in ["good", "excellent"] and earnings_stability == "high":
            overall_quality = "excellent"
        elif margin_quality == "low" or capital_efficiency == "poor" or earnings_stability == "low":
            overall_quality = "below_average"
        else:
            overall_quality = "good"
        
        return {
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "net_margin": net_margin,
            "roe": roe,
            "roce": roce,
            "roic": roic,
            "margin_quality": margin_quality,
            "capital_efficiency": capital_efficiency,
            "earnings_stability": earnings_stability,
            "overall_quality": overall_quality,
            "quality_notes": self._generate_quality_notes(gross_margin, net_margin, roe, roic, earnings_stability)
        }
    
    def _generate_quality_notes(self, gross_margin: float, net_margin: float, 
                             roe: float, roic: float, earnings_stability: str) -> str:
        """
        Generate business quality analysis notes.
        
        Args:
            gross_margin: Gross margin percentage
            net_margin: Net margin percentage
            roe: Return on equity percentage
            roic: Return on invested capital percentage
            earnings_stability: Earnings stability rating
            
        Returns:
            Quality notes string
        """
        notes = []
        
        # Margin analysis
        if gross_margin > self.analysis_params["good_gross_margin"]:
            notes.append(f"Strong gross margin of {gross_margin:.2f}% indicates pricing power and efficient production.")
        else:
            notes.append(f"Gross margin of {gross_margin:.2f}% suggests moderate pricing power and production efficiency.")
        
        if net_margin > self.analysis_params["good_net_margin"]:
            notes.append(f"High net margin of {net_margin:.2f}% demonstrates excellent overall operational efficiency.")
        else:
            notes.append(f"Net margin of {net_margin:.2f}% indicates moderate profitability.")
        
        # Return metrics analysis
        if roe > self.analysis_params["great_roe"]:
            notes.append(f"Excellent ROE of {roe:.2f}% shows superior shareholder value creation.")
        elif roe > self.analysis_params["good_roe"]:
            notes.append(f"Good ROE of {roe:.2f}% indicates solid shareholder value creation.")
        else:
            notes.append(f"ROE of {roe:.2f}% suggests moderate capital efficiency.")
        
        if roic > self.analysis_params["great_roic"]:
            notes.append(f"Outstanding ROIC of {roic:.2f}% demonstrates excellent capital allocation efficiency.")
        elif roic > self.analysis_params["good_roic"]:
            notes.append(f"Good ROIC of {roic:.2f}% indicates efficient capital allocation.")
        else:
            notes.append(f"ROIC of {roic:.2f}% suggests moderate investment returns.")
        
        # Earnings stability analysis
        if earnings_stability == "high":
            notes.append("High earnings stability indicates predictable and consistent profit generation.")
        elif earnings_stability == "low":
            notes.append("Low earnings stability suggests volatility in profit generation, which may indicate business cyclicality or inconsistent performance.")
        else:
            notes.append("Moderate earnings stability with some fluctuations in quarterly performance.")
        
        # Return combined notes
        return " ".join(notes)
    
    def _compare_with_sector(self, ratios: Dict[str, Any], growth_trends: Dict[str, Any], 
                           sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare company metrics with sector benchmarks.
        
        Args:
            ratios: Financial ratios dictionary
            growth_trends: Growth trends dictionary
            sector: Company sector
            
        Returns:
            Dictionary with sector comparison
        """
        if not sector or sector not in self.sector_benchmarks:
            sector = "default"
        
        sector_data = self.sector_benchmarks[sector]
        
        # Calculate differences from sector averages
        pe_diff = calculate_percentage_diff(ratios.get("pe_ratio", 0), sector_data["pe_ratio"])
        pb_diff = calculate_percentage_diff(ratios.get("pb_ratio", 0), sector_data["pb_ratio"])
        roe_diff = calculate_percentage_diff(ratios.get("roe", 0), sector_data["roe"])
        net_margin_diff = calculate_percentage_diff(ratios.get("net_margin", 0), sector_data["net_margin"])
        
        # Get growth rate
        avg_revenue_growth = growth_trends.get("revenue_growth", {}).get("avg_growth")
        growth_diff = calculate_percentage_diff(avg_revenue_growth, sector_data["revenue_growth"]) if avg_revenue_growth else None
        
        # Determine overall sector positioning
        strengths = []
        weaknesses = []
        
        if roe_diff and roe_diff > 20:
            strengths.append("superior_roe")
        elif roe_diff and roe_diff < -20:
            weaknesses.append("inferior_roe")
        
        if net_margin_diff and net_margin_diff > 20:
            strengths.append("superior_margins")
        elif net_margin_diff and net_margin_diff < -20:
            weaknesses.append("inferior_margins")
        
        if growth_diff and growth_diff > 20:
            strengths.append("superior_growth")
        elif growth_diff and growth_diff < -20:
            weaknesses.append("inferior_growth")
        
        if pe_diff and pe_diff < -20:
            strengths.append("attractive_valuation")
        elif pe_diff and pe_diff > 20:
            weaknesses.append("premium_valuation")
        
        # Overall sector comparison
        sector_positioning = "in_line"
        if len(strengths) > len(weaknesses) and len(strengths) >= 2:
            sector_positioning = "outperformer"
        elif len(weaknesses) > len(strengths) and len(weaknesses) >= 2:
            sector_positioning = "underperformer"
        
        return {
            "sector": sector,
            "pe_ratio": {
                "company": ratios.get("pe_ratio", 0),
                "sector": sector_data["pe_ratio"],
                "difference": pe_diff
            },
            "pb_ratio": {
                "company": ratios.get("pb_ratio", 0),
                "sector": sector_data["pb_ratio"],
                "difference": pb_diff
            },
            "roe": {
                "company": ratios.get("roe", 0),
                "sector": sector_data["roe"],
                "difference": roe_diff
            },
            "net_margin": {
                "company": ratios.get("net_margin", 0),
                "sector": sector_data["net_margin"],
                "difference": net_margin_diff
            },
            "revenue_growth": {
                "company": avg_revenue_growth,
                "sector": sector_data["revenue_growth"],
                "difference": growth_diff
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "sector_positioning": sector_positioning
        }
    
    def _generate_investment_thesis(self, company_info: Dict[str, Any], ratios: Dict[str, Any], 
                                  growth_trends: Dict[str, Any], valuation: Dict[str, Any],
                                  financial_health: Dict[str, Any], quality_metrics: Dict[str, Any], 
                                  sector_comparison: Dict[str, Any]) -> str:
        """
        Generate investment thesis.
        
        Args:
            company_info: Company information dictionary
            ratios: Financial ratios dictionary
            growth_trends: Growth trends dictionary
            valuation: Valuation analysis dictionary
            financial_health: Financial health analysis dictionary
            quality_metrics: Quality metrics dictionary
            sector_comparison: Sector comparison dictionary
            
        Returns:
            Investment thesis string
        """
        thesis_points = []
        
        # Company overview
        company_name = company_info.get("name", "The company")
        sector = company_info.get("sector", "its sector")
        thesis_points.append(f"{company_name} operates in the {sector} sector.")
        
        # Business quality
        overall_quality = quality_metrics.get("overall_quality", "good")
        if overall_quality == "excellent":
            thesis_points.append(f"The company demonstrates excellent business quality with strong margins, high capital efficiency, and consistent earnings.")
        elif overall_quality == "good":
            thesis_points.append(f"The company shows good business quality with solid profitability metrics and capital returns.")
        else:
            thesis_points.append(f"The company exhibits below-average business quality with challenges in profitability or capital efficiency.")
        
        # Growth prospects
        revenue_growth = growth_trends.get("revenue_growth", {}).get("avg_growth")
        profit_growth = growth_trends.get("profit_growth", {}).get("avg_growth")
        revenue_trend = growth_trends.get("revenue_growth", {}).get("trend")
        
        if revenue_growth and profit_growth:
            if revenue_growth > 15 and profit_growth > 20:
                thesis_points.append(f"The company has demonstrated strong growth with revenue increasing at {revenue_growth:.1f}% and profits at {profit_growth:.1f}% on average.")
            elif revenue_growth > 8 and profit_growth > 10:
                thesis_points.append(f"The company has shown solid growth with revenue increasing at {revenue_growth:.1f}% and profits at {profit_growth:.1f}% on average.")
            else:
                thesis_points.append(f"The company has experienced moderate growth with revenue increasing at {revenue_growth:.1f}% and profits at {profit_growth:.1f}% on average.")
        
        if revenue_trend:
            if revenue_trend == "accelerating":
                thesis_points.append("Revenue growth appears to be accelerating in recent quarters.")
            elif revenue_trend == "decelerating":
                thesis_points.append("Revenue growth has been decelerating in recent quarters.")
        
        # Financial health
        overall_health = financial_health.get("overall_health")
        if overall_health == "excellent":
            thesis_points.append("The company maintains excellent financial health with low debt, strong liquidity, and robust cash flow generation.")
        elif overall_health == "good":
            thesis_points.append("The company demonstrates good financial health with manageable debt levels and adequate liquidity.")
        else:
            thesis_points.append("The company shows some concerning financial health metrics, including elevated debt or liquidity constraints.")
        
        # Valuation
        valuation_status = valuation.get("status")
        if valuation_status == "undervalued":
            thesis_points.append("Current valuation metrics suggest the stock may be undervalued relative to fundamentals and growth prospects.")
        elif valuation_status == "overvalued":
            thesis_points.append("Current valuation metrics indicate the stock may be overvalued relative to fundamentals and growth prospects.")
        else:
            thesis_points.append("Current valuation metrics suggest the stock is fairly valued relative to fundamentals and growth prospects.")
        
        # Sector comparison
        sector_positioning = sector_comparison.get("sector_positioning")
        if sector_positioning == "outperformer":
            thesis_points.append(f"The company outperforms its sector peers in key metrics including profitability, growth, and capital efficiency.")
        elif sector_positioning == "underperformer":
            thesis_points.append(f"The company underperforms its sector peers in several key metrics including profitability, growth, or capital efficiency.")
        else:
            thesis_points.append(f"The company's performance is generally in line with sector averages across key metrics.")
        
        # Overall investment view
        if overall_quality in ["good", "excellent"] and overall_health in ["good", "excellent"] and (valuation_status == "undervalued" or sector_positioning == "outperformer"):
            thesis_points.append("Overall, the company presents a compelling investment opportunity given its strong fundamentals, financial health, and attractive valuation relative to prospects.")
        elif overall_quality == "below_average" or overall_health == "concerning" or valuation_status == "overvalued":
            thesis_points.append("Overall, the company presents significant investment risks that may outweigh potential rewards at current valuations.")
        else:
            thesis_points.append("Overall, the company presents a balanced investment proposition with both positive attributes and potential concerns to monitor.")
        
        # Combine thesis points
        return " ".join(thesis_points)
    
    def _generate_swot_analysis(self, company_info: Dict[str, Any], ratios: Dict[str, Any], 
                              growth_trends: Dict[str, Any], valuation: Dict[str, Any],
                              financial_health: Dict[str, Any], quality_metrics: Dict[str, Any], 
                              sector_comparison: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate SWOT analysis.
        
        Args:
            company_info: Company information dictionary
            ratios: Financial ratios dictionary
            growth_trends: Growth trends dictionary
            valuation: Valuation analysis dictionary
            financial_health: Financial health analysis dictionary
            quality_metrics: Quality metrics dictionary
            sector_comparison: Sector comparison dictionary
            
        Returns:
            Dictionary with SWOT analysis
        """
        strengths = []
        weaknesses = []
        opportunities = []
        threats = []
        
        # Strengths
        if quality_metrics.get("margin_quality") == "high":
            strengths.append(f"Strong margins (Gross: {quality_metrics.get('gross_margin', 0):.1f}%, Net: {quality_metrics.get('net_margin', 0):.1f}%)")
        
        if quality_metrics.get("capital_efficiency") in ["good", "excellent"]:
            strengths.append(f"High capital efficiency (ROE: {quality_metrics.get('roe', 0):.1f}%, ROIC: {quality_metrics.get('roic', 0):.1f}%)")
        
        if quality_metrics.get("earnings_stability") == "high":
            strengths.append("Consistent and stable earnings generation")
        
        if financial_health.get("debt_level") == "low":
            strengths.append(f"Low financial leverage (Debt/Equity: {financial_health.get('debt_equity', 0):.2f})")
        
        if financial_health.get("liquidity_status") == "strong":
            strengths.append(f"Strong liquidity position (Current Ratio: {financial_health.get('current_ratio', 0):.2f})")
        
        growth_rate = growth_trends.get("revenue_growth", {}).get("avg_growth")
        if growth_rate and growth_rate > 15:
            strengths.append(f"Strong revenue growth ({growth_rate:.1f}% on average)")
        
        # Add sector strengths
        for strength in sector_comparison.get("strengths", []):
            if strength == "superior_roe":
                strengths.append("Superior ROE compared to sector peers")
            elif strength == "superior_margins":
                strengths.append("Higher profit margins than sector average")
            elif strength == "superior_growth":
                strengths.append("Faster growth than sector average")
            elif strength == "attractive_valuation":
                strengths.append("Attractive valuation relative to sector")
        
        # Weaknesses
        if quality_metrics.get("margin_quality") == "low":
            weaknesses.append(f"Weak margins (Gross: {quality_metrics.get('gross_margin', 0):.1f}%, Net: {quality_metrics.get('net_margin', 0):.1f}%)")
        
        if quality_metrics.get("capital_efficiency") == "poor":
            weaknesses.append(f"Poor capital efficiency (ROE: {quality_metrics.get('roe', 0):.1f}%, ROIC: {quality_metrics.get('roic', 0):.1f}%)")
        
        if quality_metrics.get("earnings_stability") == "low":
            weaknesses.append("Volatile earnings with significant quarter-to-quarter fluctuations")
        
        if financial_health.get("debt_level") == "high":
            weaknesses.append(f"High financial leverage (Debt/Equity: {financial_health.get('debt_equity', 0):.2f})")
        
        if financial_health.get("liquidity_status") == "poor":
            weaknesses.append(f"Poor liquidity position (Current Ratio: {financial_health.get('current_ratio', 0):.2f})")
        
        if growth_rate and growth_rate < 5:
            weaknesses.append(f"Slow revenue growth ({growth_rate:.1f}% on average)")
        
        # Add sector weaknesses
        for weakness in sector_comparison.get("weaknesses", []):
            if weakness == "inferior_roe":
                weaknesses.append("Lower ROE compared to sector peers")
            elif weakness == "inferior_margins":
                weaknesses.append("Lower profit margins than sector average")
            elif weakness == "inferior_growth":
                weaknesses.append("Slower growth than sector average")
            elif weakness == "premium_valuation":
                weaknesses.append("Premium valuation relative to sector")
        
        # Opportunities
        if growth_trends.get("revenue_growth", {}).get("trend") == "accelerating":
            opportunities.append("Accelerating revenue growth trend")
        
        if growth_trends.get("margin_trends", {}).get("trend") == "improving":
            opportunities.append("Improving margin trend")
        
        if valuation.get("status") == "undervalued":
            opportunities.append("Potential for valuation multiple expansion")
        
        # Generic opportunities
        opportunities.append("Potential for market share gains in core business segments")
        opportunities.append("Expansion opportunities in adjacent markets or product categories")
        
        if sector_comparison.get("sector_positioning") != "outperformer":
            opportunities.append("Room to improve performance relative to sector peers")
        
        # Threats
        if growth_trends.get("revenue_growth", {}).get("trend") == "decelerating":
            threats.append("Decelerating revenue growth trend")
        
        if growth_trends.get("margin_trends", {}).get("trend") == "declining":
            threats.append("Declining margin trend")
        
        if valuation.get("status") == "overvalued":
            threats.append("Risk of valuation multiple contraction")
        
        if financial_health.get("debt_level") == "high":
            threats.append("Vulnerability to interest rate increases")
        
        # Generic threats
        threats.append("Competitive pressure in core markets")
        threats.append("Potential regulatory changes affecting the industry")
        threats.append("Macroeconomic sensitivity and cyclical headwinds")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "opportunities": opportunities,
            "threats": threats
        }
    
    def _calculate_fundamental_score(self, ratios: Dict[str, Any], growth_trends: Dict[str, Any], 
                                   valuation: Dict[str, Any], financial_health: Dict[str, Any],
                                   quality_metrics: Dict[str, Any], 
                                   sector_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate fundamental score across different categories.
        
        Args:
            ratios: Financial ratios dictionary
            growth_trends: Growth trends dictionary
            valuation: Valuation analysis dictionary
            financial_health: Financial health analysis dictionary
            quality_metrics: Quality metrics dictionary
            sector_comparison: Sector comparison dictionary
            
        Returns:
            Dictionary with fundamental scores
        """
        # Calculate profitability score (0-100)
        profitability_score = 0
        
        # ROE component (0-40)
        # ROE component (0-40)
        roe = quality_metrics.get("roe", 0)
        if roe > self.analysis_params["great_roe"]:
            profitability_score += 40
        elif roe > self.analysis_params["good_roe"]:
            profitability_score += 25 + (roe - self.analysis_params["good_roe"]) / (self.analysis_params["great_roe"] - self.analysis_params["good_roe"]) * 15
        elif roe > 0:
            profitability_score += max(0, 25 * roe / self.analysis_params["good_roe"])
        
        # Net margin component (0-40)
        net_margin = quality_metrics.get("net_margin", 0)
        if net_margin > self.analysis_params["good_net_margin"] * 1.5:
            margin_score = 40
        elif net_margin > self.analysis_params["good_net_margin"]:
            margin_score = 25 + (net_margin - self.analysis_params["good_net_margin"]) / (self.analysis_params["good_net_margin"] * 0.5) * 15
        elif net_margin > 0:
            margin_score = max(0, 25 * net_margin / self.analysis_params["good_net_margin"])
        else:
            margin_score = 0
        
        profitability_score += margin_score
        
        # Normalize to 0-100
        profitability_score = min(100, profitability_score)
        
        # Calculate growth score (0-100)
        growth_score = 0
        
        # Revenue growth component (0-50)
        revenue_growth = growth_trends.get("revenue_growth", {}).get("avg_growth", 0)
        if revenue_growth > self.analysis_params["strong_revenue_growth"]:
            growth_score += 50
        elif revenue_growth > 0:
            growth_score += min(45, revenue_growth / self.analysis_params["strong_revenue_growth"] * 50)
        
        # Profit growth component (0-50)
        profit_growth = growth_trends.get("profit_growth", {}).get("avg_growth", 0)
        if profit_growth > self.analysis_params["strong_profit_growth"]:
            growth_score += 50
        elif profit_growth > 0:
            growth_score += min(45, profit_growth / self.analysis_params["strong_profit_growth"] * 50)
        
        # Normalize to 0-100
        growth_score = min(100, growth_score)
        
        # Calculate financial health score (0-100)
        health_score = 0
        
        # Debt level component (0-40)
        debt_equity = financial_health.get("debt_equity", 0)
        if debt_equity < self.analysis_params["safe_debt_equity"]:
            debt_score = 40
        elif debt_equity < self.analysis_params["concerning_debt_equity"]:
            debt_score = 40 - (debt_equity - self.analysis_params["safe_debt_equity"]) / (self.analysis_params["concerning_debt_equity"] - self.analysis_params["safe_debt_equity"]) * 30
        else:
            debt_score = max(0, 10 - (debt_equity - self.analysis_params["concerning_debt_equity"]) * 2)
        
        health_score += debt_score
        
        # Liquidity component (0-30)
        current_ratio = financial_health.get("current_ratio", 0)
        if current_ratio > self.analysis_params["good_current_ratio"]:
            liquidity_score = 30
        elif current_ratio > 1.0:
            liquidity_score = 15 + (current_ratio - 1.0) / (self.analysis_params["good_current_ratio"] - 1.0) * 15
        else:
            liquidity_score = max(0, current_ratio * 15)
        
        health_score += liquidity_score
        
        # Interest coverage component (0-30)
        interest_coverage = financial_health.get("interest_coverage", 0)
        if interest_coverage > self.analysis_params["good_interest_coverage"]:
            interest_score = 30
        elif interest_coverage > 2.0:
            interest_score = 15 + (interest_coverage - 2.0) / (self.analysis_params["good_interest_coverage"] - 2.0) * 15
        elif interest_coverage > 0:
            interest_score = max(0, interest_coverage / 2.0 * 15)
        else:
            interest_score = 0
        
        health_score += interest_score
        
        # Normalize to 0-100
        health_score = min(100, health_score)
        
        # Calculate valuation score (0-100)
        valuation_score = 0
        
        # PE ratio component (0-50)
        pe_ratio = valuation.get("pe_ratio", 0)
        sector_pe = valuation.get("sector_pe", 20)
        
        if pe_ratio <= 0:  # Negative PE (negative earnings)
            pe_score = 0
        elif pe_ratio < sector_pe * 0.7:
            pe_score = 50  # Significantly below sector average
        elif pe_ratio < sector_pe:
            pe_score = 35 + (sector_pe - pe_ratio) / (sector_pe * 0.3) * 15
        elif pe_ratio < sector_pe * 1.3:
            pe_score = 35 - (pe_ratio - sector_pe) / (sector_pe * 0.3) * 15
        else:
            pe_score = max(0, 20 - (pe_ratio - sector_pe * 1.3) / sector_pe * 20)
        
        valuation_score += pe_score
        
        # PEG ratio component (0-50)
        peg_ratio = valuation.get("peg_ratio")
        if peg_ratio:
            if peg_ratio < 1.0:
                peg_score = 50
            elif peg_ratio < 2.0:
                peg_score = 50 - (peg_ratio - 1.0) * 25
            else:
                peg_score = max(0, 25 - (peg_ratio - 2.0) * 10)
            
            valuation_score += peg_score
        else:
            # If PEG ratio not available, use sector comparison
            pe_diff = sector_comparison.get("pe_ratio", {}).get("difference", 0)
            
            if pe_diff < -20:  # PE is >20% below sector
                valuation_score += 40
            elif pe_diff < 0:
                valuation_score += 30 + pe_diff / 20 * 10
            elif pe_diff < 20:
                valuation_score += 30 - pe_diff / 20 * 10
            else:
                valuation_score += max(0, 20 - (pe_diff - 20) / 10)
        
        # Normalize to 0-100
        valuation_score = min(100, valuation_score)
        
        # Calculate overall score (weighted average)
        weights = {
            "profitability": 0.25,
            "growth": 0.25,
            "financial_health": 0.2,
            "valuation": 0.3
        }
        
        overall_score = (
            profitability_score * weights["profitability"] +
            growth_score * weights["growth"] +
            health_score * weights["financial_health"] +
            valuation_score * weights["valuation"]
        )
        
        # Determine rating based on overall score
        if overall_score >= 80:
            rating = "Strong Buy"
        elif overall_score >= 65:
            rating = "Buy"
        elif overall_score >= 50:
            rating = "Hold"
        elif overall_score >= 35:
            rating = "Reduce"
        else:
            rating = "Sell"
        
        return {
            "profitability": {
                "score": int(profitability_score),
                "rating": self._get_category_rating(profitability_score)
            },
            "growth": {
                "score": int(growth_score),
                "rating": self._get_category_rating(growth_score)
            },
            "financial_health": {
                "score": int(health_score),
                "rating": self._get_category_rating(health_score)
            },
            "valuation": {
                "score": int(valuation_score),
                "rating": self._get_category_rating(valuation_score)
            },
            "overall": {
                "score": int(overall_score),
                "rating": rating
            }
        }
    
    def _get_category_rating(self, score: float) -> str:
        """
        Get rating text based on category score.
        
        Args:
            score: Numerical score (0-100)
            
        Returns:
            Rating text
        """
        if score >= 80:
            return "Excellent"
        elif score >= 65:
            return "Good"
        elif score >= 50:
            return "Average"
        elif score >= 35:
            return "Below Average"
        else:
            return "Poor"
    
    def _generate_recommendation(self, fundamental_score: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate investment recommendation based on fundamental score.
        
        Args:
            fundamental_score: Fundamental score dictionary
            
        Returns:
            Dictionary with recommendation details
        """
        # Extract overall rating
        overall_rating = fundamental_score.get("overall", {}).get("rating", "Hold")
        overall_score = fundamental_score.get("overall", {}).get("score", 50)
        
        # Generate recommendation text
        if overall_rating == "Strong Buy":
            recommendation_text = "The company demonstrates exceptional fundamentals across profitability, growth, financial health, and valuation. Consider adding to core long-term holdings."
        elif overall_rating == "Buy":
            recommendation_text = "The company shows solid fundamentals with particular strengths in key areas. Represents an attractive investment at current levels."
        elif overall_rating == "Hold":
            recommendation_text = "The company exhibits a balanced mix of strengths and weaknesses. Current shareholders should maintain positions, while new investors may wait for more compelling entry points."
        elif overall_rating == "Reduce":
            recommendation_text = "The company faces significant challenges in key fundamental areas. Consider reducing exposure or implementing tight risk management."
        else:  # Sell
            recommendation_text = "The company shows concerning fundamentals across multiple dimensions. The risk-reward ratio appears unfavorable at current levels."
        
        # Generate key reasons based on category scores
        key_reasons = []
        
        # Find strongest category
        categories = ["profitability", "growth", "financial_health", "valuation"]
        scores = [fundamental_score.get(c, {}).get("score", 0) for c in categories]
        strongest_idx = scores.index(max(scores))
        strongest_category = categories[strongest_idx]
        
        # Find weakest category
        weakest_idx = scores.index(min(scores))
        weakest_category = categories[weakest_idx]
        
        # Add reasons based on strongest and weakest categories
        if fundamental_score.get(strongest_category, {}).get("score", 0) >= 65:
            if strongest_category == "profitability":
                key_reasons.append("Strong profitability metrics with excellent returns on capital")
            elif strongest_category == "growth":
                key_reasons.append("Impressive growth trajectory in revenue and earnings")
            elif strongest_category == "financial_health":
                key_reasons.append("Rock-solid financial position with low debt and strong liquidity")
            elif strongest_category == "valuation":
                key_reasons.append("Attractive valuation relative to fundamentals and growth prospects")
        
        if fundamental_score.get(weakest_category, {}).get("score", 0) < 50:
            if weakest_category == "profitability":
                key_reasons.append("Concerning profitability metrics with below-average returns")
            elif weakest_category == "growth":
                key_reasons.append("Weak or inconsistent growth in revenue and earnings")
            elif weakest_category == "financial_health":
                key_reasons.append("Financial position shows vulnerabilities in debt levels or liquidity")
            elif weakest_category == "valuation":
                key_reasons.append("Valuation appears stretched relative to fundamentals")
        
        # Add an overall reason based on score
        if overall_score >= 75:
            key_reasons.append("Overall, the company demonstrates superior fundamental quality across multiple dimensions")
        elif overall_score >= 60:
            key_reasons.append("The company's strengths outweigh its weaknesses in key fundamental areas")
        elif overall_score >= 45:
            key_reasons.append("The company shows a relatively balanced mix of strengths and weaknesses")
        elif overall_score >= 30:
            key_reasons.append("Weaknesses in key fundamental areas outweigh the company's strengths")
        else:
            key_reasons.append("Significant fundamental challenges across multiple critical dimensions")
        
        # Timeframe recommendation
        if overall_rating in ["Strong Buy", "Buy"]:
            timeframe = "Long-term potential with favorable risk-reward ratio"
        elif overall_rating == "Hold":
            timeframe = "Monitor for changes in fundamentals or more attractive entry points"
        else:
            timeframe = "Consider exit or reduction in position size"
        
        # Risk level
        risk_level = "moderate"
        financial_health_score = fundamental_score.get("financial_health", {}).get("score", 50)
        profitability_score = fundamental_score.get("profitability", {}).get("score", 50)
        
        if financial_health_score >= 70 and profitability_score >= 70:
            risk_level = "low"
        elif financial_health_score < 40 or profitability_score < 40:
            risk_level = "high"
        
        return {
            "rating": overall_rating,
            "recommendation": recommendation_text,
            "key_reasons": key_reasons,
            "timeframe": timeframe,
            "risk_level": risk_level
        }
    
    def _save_analysis(self, symbol: str, exchange: str, result: Dict[str, Any]) -> bool:
        """
        Save analysis result to database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            result: Analysis result
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare document for storage
            document = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now(),
                "company_info": result.get("company_info"),
                "financial_summary": result.get("financial_summary"),
                "sector_comparison": result.get("sector_comparison"),
                "peer_comparison": result.get("peer_comparison"),
                "investment_thesis": result.get("investment_thesis"),
                "swot_analysis": result.get("swot_analysis"),
                "fundamental_score": result.get("fundamental_score"),
                "recommendation": result.get("recommendation")
            }
            
            # Insert into database
            result = self.db.fundamental_analysis_collection.insert_one(document)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving analysis result: {e}")
            return False
    
    def screen_stocks(self, criteria: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Screen stocks based on fundamental criteria.
        
        Args:
            criteria: Dictionary with screening criteria
            limit: Maximum number of results to return
            
        Returns:
            List of stocks meeting the criteria
        """
        try:
            query = {}
            
            # Process criteria
            if "min_roe" in criteria:
                query["financial_summary.key_ratios.roe"] = {"$gte": criteria["min_roe"]}
            
            if "max_pe" in criteria:
                query["financial_summary.key_ratios.pe_ratio"] = {"$lte": criteria["max_pe"]}
            
            if "min_growth" in criteria:
                query["financial_summary.growth_trends.revenue_growth.avg_growth"] = {"$gte": criteria["min_growth"]}
            
            if "max_debt_equity" in criteria:
                query["financial_summary.financial_health.debt_equity"] = {"$lte": criteria["max_debt_equity"]}
            
            if "min_net_margin" in criteria:
                query["financial_summary.key_ratios.net_margin"] = {"$gte": criteria["min_net_margin"]}
            
            if "min_score" in criteria:
                query["fundamental_score.overall.score"] = {"$gte": criteria["min_score"]}
            
            if "sector" in criteria:
                query["company_info.sector"] = criteria["sector"]
            
            # Execute query
            cursor = self.db.fundamental_analysis_collection.find(
                query,
                {
                    "symbol": 1,
                    "exchange": 1,
                    "company_info.name": 1,
                    "company_info.sector": 1,
                    "financial_summary.key_ratios": 1,
                    "financial_summary.growth_trends.revenue_growth.avg_growth": 1,
                    "financial_summary.financial_health.debt_equity": 1,
                    "fundamental_score.overall": 1,
                    "recommendation.rating": 1
                }
            ).sort("fundamental_score.overall.score", -1).limit(limit)
            
            results = list(cursor)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error screening stocks: {e}")
            return []


def calculate_growth(current: float, previous: float) -> Optional[float]:
    """
    Calculate percentage growth between two values.
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Growth percentage or None if invalid
    """
    if previous is None or current is None:
        return None
    
    if previous == 0:
        return None
    
    return ((current / previous) - 1) * 100


def calculate_percentage_diff(value1: float, value2: float) -> Optional[float]:
    """
    Calculate percentage difference between two values.
    
    Args:
        value1: First value
        value2: Second value
        
    Returns:
        Percentage difference or None if invalid
    """
    if value1 is None or value2 is None or value1 == 0 or value2 == 0:
        return None
    
    return ((value1 / value2) - 1) * 100

def analyze_ratio_trend(self, symbol: str, exchange: str, ratio_name: str, 
                       quarters: int = 8) -> Dict[str, Any]:
    """
    Analyze trend of a specific financial ratio over time.
    
    Args:
        symbol: Stock symbol
        exchange: Stock exchange
        ratio_name: Name of ratio to analyze
        quarters: Number of quarters to analyze
        
    Returns:
        Dictionary with ratio trend analysis
    """
    try:
        self.logger.info(f"Analyzing {ratio_name} trend for {symbol} ({exchange})")
        
        # Get quarterly results
        cursor = self.db.quarterly_results_collection.find({
            "symbol": symbol,
            "exchange": exchange
        }).sort("report_date", -1).limit(quarters)
        
        quarterly_results = list(cursor)
        
        if not quarterly_results:
            return {
                "symbol": symbol,
                "exchange": exchange,
                "ratio": ratio_name,
                "status": "error",
                "error": "Quarterly results not found"
            }
        
        # Extract ratio values
        ratio_values = []
        
        for quarter in quarterly_results:
            # Handle different ratio locations in the document
            if ratio_name in quarter:
                ratio_values.append({
                    "period": quarter.get("period"),
                    "value": quarter[ratio_name]
                })
            elif "key_ratios" in quarter and ratio_name in quarter["key_ratios"]:
                ratio_values.append({
                    "period": quarter.get("period"),
                    "value": quarter["key_ratios"][ratio_name]
                })
        
        # Sort by period
        ratio_values.reverse()  # To get chronological order
        
        # Calculate trend
        values = [v["value"] for v in ratio_values if v["value"] is not None]
        trend = determine_trend(values)
        
        # Calculate statistics
        mean = np.mean(values) if values else None
        median = np.median(values) if values else None
        std_dev = np.std(values) if len(values) > 1 else None
        
        # Calculate linear regression
        slope = None
        intercept = None
        r_squared = None
        
        if len(values) > 2:
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            ss_res = np.sum((values - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Generate insights
        insights = []
        
        if trend == "accelerating" and ratio_name in ["roe", "roce", "net_margin", "operating_margin"]:
            insights.append(f"{ratio_name.upper()} is showing an improving trend, which indicates strengthening business performance.")
        elif trend == "decelerating" and ratio_name in ["roe", "roce", "net_margin", "operating_margin"]:
            insights.append(f"{ratio_name.upper()} is showing a deteriorating trend, which may indicate challenges in the business model.")
        
        if ratio_name in ["debt_equity", "debt_to_equity"]:
            if trend == "accelerating":
                insights.append("Debt levels are increasing, which may increase financial risk.")
            elif trend == "decelerating":
                insights.append("Debt levels are decreasing, which suggests improving financial health.")
        
        # Return analysis
        return {
            "symbol": symbol,
            "exchange": exchange,
            "ratio": ratio_name,
            "status": "success",
            "values": ratio_values,
            "trend": trend,
            "statistics": {
                "mean": mean,
                "median": median,
                "std_dev": std_dev
            },
            "regression": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared
            },
            "insights": insights
        }
        
    except Exception as e:
        self.logger.error(f"Error analyzing {ratio_name} trend for {symbol}: {e}")
        return {
            "symbol": symbol,
            "exchange": exchange,
            "ratio": ratio_name,
            "status": "error",
            "error": str(e)
        }

def determine_trend(values: List[float]) -> str:
    """
    Determine trend based on series of values.
    
    Args:
        values: List of values
        
    Returns:
        Trend description
    """
    if not values or len(values) < 3:
        return "stable"
    
    # Linear regression
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    
    # Calculate average to normalize slope
    avg = np.mean(values)
    normalized_slope = slope / avg if avg != 0 else slope
    
    if normalized_slope > 0.05:
        return "accelerating"
    elif normalized_slope < -0.05:
        return "decelerating"
    else:
        return "stable"