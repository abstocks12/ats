"""
Fundamental Metrics Calculator

This module calculates standard fundamental financial metrics from data collected by the financial scraper.
It processes raw financial data into standardized metrics used across the trading system.
"""

import numpy as np
import pandas as pd
import math
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

class FundamentalMetrics:
    """
    Calculates standardized fundamental financial metrics from raw financial data.
    Provides methods to process, normalize, and derive financial indicators.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the fundamental metrics calculator with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
    
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
    
    def process_raw_financial_data(self, financial_data: Dict[str, Any], symbol: str, 
                                   exchange: str) -> Dict[str, Any]:
        """
        Process raw financial data from the scraper and calculate standardized metrics.
        
        Args:
            financial_data: Raw financial data from scraper
            symbol: Stock symbol/ticker
            exchange: Stock exchange
            
        Returns:
            Dictionary containing standardized fundamental metrics
        """
        self.logger.info(f"Processing raw financial data for {symbol}:{exchange}")
        
        try:
            # Extract and calculate metrics
            income_statement_metrics = self._process_income_statement(financial_data)
            balance_sheet_metrics = self._process_balance_sheet(financial_data)
            cash_flow_metrics = self._process_cash_flow(financial_data)
            per_share_metrics = self._calculate_per_share_metrics(
                financial_data, income_statement_metrics, balance_sheet_metrics, cash_flow_metrics
            )
            ratio_metrics = self._calculate_financial_ratios(
                income_statement_metrics, balance_sheet_metrics, cash_flow_metrics, per_share_metrics
            )
            
            # Get market data for valuation metrics
            market_price = self._get_current_market_price(symbol, exchange)
            valuation_metrics = self._calculate_valuation_metrics(
                per_share_metrics, market_price, financial_data
            )
            
            # Calculate growth metrics
            growth_metrics = self._calculate_growth_metrics(financial_data)
            
            # Combine all metrics
            combined_metrics = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now(),
                "sector": financial_data.get("sector", ""),
                "industry": financial_data.get("industry", ""),
                "income_statement": income_statement_metrics,
                "balance_sheet": balance_sheet_metrics,
                "cash_flow": cash_flow_metrics,
                "per_share": per_share_metrics,
                "ratios": ratio_metrics,
                "valuation": valuation_metrics,
                "growth": growth_metrics
            }
            
            # Save to database
            self._save_to_database(combined_metrics)
            
            return combined_metrics
            
        except Exception as e:
            self.logger.error(f"Error processing financial data: {str(e)}")
            return {"error": str(e), "symbol": symbol, "exchange": exchange}
    
    def _process_income_statement(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize income statement metrics."""
        metrics = {}
        
        # Check for income statement data
        income_stmt = financial_data.get("income_statement", {})
        
        # If no dedicated income statement section, try to find relevant fields
        if not income_stmt:
            income_stmt = financial_data
        
        # Extract revenue/sales (different naming conventions)
        if "revenue" in income_stmt:
            metrics["revenue"] = income_stmt["revenue"]
        elif "sales" in income_stmt:
            metrics["revenue"] = income_stmt["sales"]
        elif "total_revenue" in income_stmt:
            metrics["revenue"] = income_stmt["total_revenue"]
        
        # Cost of Revenue / COGS
        if "cost_of_revenue" in income_stmt:
            metrics["cost_of_revenue"] = income_stmt["cost_of_revenue"]
        elif "cogs" in income_stmt:
            metrics["cost_of_revenue"] = income_stmt["cogs"]
        elif "cost_of_goods_sold" in income_stmt:
            metrics["cost_of_revenue"] = income_stmt["cost_of_goods_sold"]
        
        # Gross Profit
        if "gross_profit" in income_stmt:
            metrics["gross_profit"] = income_stmt["gross_profit"]
        elif "revenue" in metrics and "cost_of_revenue" in metrics:
            metrics["gross_profit"] = metrics["revenue"] - metrics["cost_of_revenue"]
        
        # Operating Expenses
        if "operating_expenses" in income_stmt:
            metrics["operating_expenses"] = income_stmt["operating_expenses"]
        elif "total_operating_expenses" in income_stmt:
            metrics["operating_expenses"] = income_stmt["total_operating_expenses"]
        
        # Operating Income / EBIT
        if "operating_income" in income_stmt:
            metrics["operating_income"] = income_stmt["operating_income"]
        elif "ebit" in income_stmt:
            metrics["operating_income"] = income_stmt["ebit"]
        elif "gross_profit" in metrics and "operating_expenses" in metrics:
            metrics["operating_income"] = metrics["gross_profit"] - metrics["operating_expenses"]
        
        # Interest Expense
        if "interest_expense" in income_stmt:
            metrics["interest_expense"] = income_stmt["interest_expense"]
        
        # Income Before Tax
        if "income_before_tax" in income_stmt:
            metrics["income_before_tax"] = income_stmt["income_before_tax"]
        elif "pre_tax_income" in income_stmt:
            metrics["income_before_tax"] = income_stmt["pre_tax_income"]
        
        # Income Tax Expense
        if "income_tax_expense" in income_stmt:
            metrics["income_tax_expense"] = income_stmt["income_tax_expense"]
        elif "tax_expense" in income_stmt:
            metrics["income_tax_expense"] = income_stmt["tax_expense"]
        
        # Net Income
        if "net_income" in income_stmt:
            metrics["net_income"] = income_stmt["net_income"]
        elif "net_profit" in income_stmt:
            metrics["net_income"] = income_stmt["net_profit"]
        
        # EBITDA
        if "ebitda" in income_stmt:
            metrics["ebitda"] = income_stmt["ebitda"]
        elif "operating_income" in metrics and "depreciation_amortization" in income_stmt:
            metrics["ebitda"] = metrics["operating_income"] + income_stmt["depreciation_amortization"]
        
        # Depreciation & Amortization
        if "depreciation_amortization" in income_stmt:
            metrics["depreciation_amortization"] = income_stmt["depreciation_amortization"]
        elif "depreciation" in income_stmt and "amortization" in income_stmt:
            metrics["depreciation_amortization"] = income_stmt["depreciation"] + income_stmt["amortization"]
        
        return metrics
    
    def _process_balance_sheet(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize balance sheet metrics."""
        metrics = {}
        
        # Check for balance sheet data
        balance_sheet = financial_data.get("balance_sheet", {})
        
        # If no dedicated balance sheet section, try to find relevant fields
        if not balance_sheet:
            balance_sheet = financial_data
        
        # Current Assets
        if "current_assets" in balance_sheet:
            metrics["current_assets"] = balance_sheet["current_assets"]
        elif "total_current_assets" in balance_sheet:
            metrics["current_assets"] = balance_sheet["total_current_assets"]
        
        # Cash and Cash Equivalents
        if "cash_and_equivalents" in balance_sheet:
            metrics["cash_and_equivalents"] = balance_sheet["cash_and_equivalents"]
        elif "cash" in balance_sheet:
            metrics["cash_and_equivalents"] = balance_sheet["cash"]
        
        # Inventory
        if "inventory" in balance_sheet:
            metrics["inventory"] = balance_sheet["inventory"]
        
        # Accounts Receivable
        if "accounts_receivable" in balance_sheet:
            metrics["accounts_receivable"] = balance_sheet["accounts_receivable"]
        elif "receivables" in balance_sheet:
            metrics["accounts_receivable"] = balance_sheet["receivables"]
        
        # Non-Current Assets
        if "non_current_assets" in balance_sheet:
            metrics["non_current_assets"] = balance_sheet["non_current_assets"]
        elif "total_non_current_assets" in balance_sheet:
            metrics["non_current_assets"] = balance_sheet["total_non_current_assets"]
        
        # Property, Plant, and Equipment (PPE)
        if "ppe" in balance_sheet:
            metrics["ppe"] = balance_sheet["ppe"]
        elif "property_plant_equipment" in balance_sheet:
            metrics["ppe"] = balance_sheet["property_plant_equipment"]
        
        # Total Assets
        if "total_assets" in balance_sheet:
            metrics["total_assets"] = balance_sheet["total_assets"]
        elif "current_assets" in metrics and "non_current_assets" in metrics:
            metrics["total_assets"] = metrics["current_assets"] + metrics["non_current_assets"]
        
        # Current Liabilities
        if "current_liabilities" in balance_sheet:
            metrics["current_liabilities"] = balance_sheet["current_liabilities"]
        elif "total_current_liabilities" in balance_sheet:
            metrics["current_liabilities"] = balance_sheet["total_current_liabilities"]
        
        # Accounts Payable
        if "accounts_payable" in balance_sheet:
            metrics["accounts_payable"] = balance_sheet["accounts_payable"]
        elif "payables" in balance_sheet:
            metrics["accounts_payable"] = balance_sheet["payables"]
        
        # Short-term Debt
        if "short_term_debt" in balance_sheet:
            metrics["short_term_debt"] = balance_sheet["short_term_debt"]
        
        # Non-Current Liabilities
        if "non_current_liabilities" in balance_sheet:
            metrics["non_current_liabilities"] = balance_sheet["non_current_liabilities"]
        elif "total_non_current_liabilities" in balance_sheet:
            metrics["non_current_liabilities"] = balance_sheet["total_non_current_liabilities"]
        
        # Long-term Debt
        if "long_term_debt" in balance_sheet:
            metrics["long_term_debt"] = balance_sheet["long_term_debt"]
        
        # Total Liabilities
        if "total_liabilities" in balance_sheet:
            metrics["total_liabilities"] = balance_sheet["total_liabilities"]
        elif "current_liabilities" in metrics and "non_current_liabilities" in metrics:
            metrics["total_liabilities"] = metrics["current_liabilities"] + metrics["non_current_liabilities"]
        
        # Shareholders' Equity
        if "shareholders_equity" in balance_sheet:
            metrics["shareholders_equity"] = balance_sheet["shareholders_equity"]
        elif "total_equity" in balance_sheet:
            metrics["shareholders_equity"] = balance_sheet["total_equity"]
        elif "total_assets" in metrics and "total_liabilities" in metrics:
            metrics["shareholders_equity"] = metrics["total_assets"] - metrics["total_liabilities"]
        
        # Working Capital
        if "current_assets" in metrics and "current_liabilities" in metrics:
            metrics["working_capital"] = metrics["current_assets"] - metrics["current_liabilities"]
        
        # Total Debt
        if "short_term_debt" in metrics and "long_term_debt" in metrics:
            metrics["total_debt"] = metrics["short_term_debt"] + metrics["long_term_debt"]
        elif "total_debt" in balance_sheet:
            metrics["total_debt"] = balance_sheet["total_debt"]
        
        # Net Debt (Total Debt - Cash)
        if "total_debt" in metrics and "cash_and_equivalents" in metrics:
            metrics["net_debt"] = metrics["total_debt"] - metrics["cash_and_equivalents"]
        
        return metrics
    
    def _process_cash_flow(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize cash flow statement metrics."""
        metrics = {}
        
        # Check for cash flow data
        cash_flow = financial_data.get("cash_flow", {})
        
        # If no dedicated cash flow section, try to find relevant fields
        if not cash_flow:
            cash_flow = financial_data
        
        # Operating Cash Flow
        if "operating_cash_flow" in cash_flow:
            metrics["operating_cash_flow"] = cash_flow["operating_cash_flow"]
        elif "cash_from_operations" in cash_flow:
            metrics["operating_cash_flow"] = cash_flow["cash_from_operations"]
        
        # Capital Expenditure
        if "capital_expenditure" in cash_flow:
            metrics["capital_expenditure"] = cash_flow["capital_expenditure"]
        elif "capex" in cash_flow:
            metrics["capital_expenditure"] = cash_flow["capex"]
        
        # Free Cash Flow
        if "free_cash_flow" in cash_flow:
            metrics["free_cash_flow"] = cash_flow["free_cash_flow"]
        elif "operating_cash_flow" in metrics and "capital_expenditure" in metrics:
            metrics["free_cash_flow"] = metrics["operating_cash_flow"] - metrics["capital_expenditure"]
        
        # Cash from Investing
        if "cash_from_investing" in cash_flow:
            metrics["cash_from_investing"] = cash_flow["cash_from_investing"]
        
        # Cash from Financing
        if "cash_from_financing" in cash_flow:
            metrics["cash_from_financing"] = cash_flow["cash_from_financing"]
        
        # Dividends Paid
        if "dividends_paid" in cash_flow:
            metrics["dividends_paid"] = cash_flow["dividends_paid"]
        
        return metrics
    
    def _calculate_per_share_metrics(self, financial_data: Dict[str, Any], 
                                    income_metrics: Dict[str, Any],
                                    balance_metrics: Dict[str, Any],
                                    cash_flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate per-share financial metrics."""
        metrics = {}
        
        # Get shares outstanding
        shares_outstanding = financial_data.get("shares_outstanding", 0)
        
        if shares_outstanding <= 0:
            return metrics  # Cannot calculate per-share metrics without shares outstanding
        
        # Earnings Per Share (EPS)
        if "net_income" in income_metrics:
            metrics["eps"] = income_metrics["net_income"] / shares_outstanding
        
        # Book Value Per Share
        if "shareholders_equity" in balance_metrics:
            metrics["book_value_per_share"] = balance_metrics["shareholders_equity"] / shares_outstanding
        
        # Revenue Per Share
        if "revenue" in income_metrics:
            metrics["revenue_per_share"] = income_metrics["revenue"] / shares_outstanding
        
        # Free Cash Flow Per Share
        if "free_cash_flow" in cash_flow_metrics:
            metrics["fcf_per_share"] = cash_flow_metrics["free_cash_flow"] / shares_outstanding
        
        # Operating Cash Flow Per Share
        if "operating_cash_flow" in cash_flow_metrics:
            metrics["ocf_per_share"] = cash_flow_metrics["operating_cash_flow"] / shares_outstanding
        
        # Dividend Per Share
        if "dividend_per_share" in financial_data:
            metrics["dividend_per_share"] = financial_data["dividend_per_share"]
        elif "dividends_paid" in cash_flow_metrics and cash_flow_metrics["dividends_paid"] < 0:
            # Dividend is usually recorded as negative in cash flow (cash outflow)
            metrics["dividend_per_share"] = abs(cash_flow_metrics["dividends_paid"]) / shares_outstanding
        
        return metrics
    
    def _calculate_financial_ratios(self, income_metrics: Dict[str, Any],
                                   balance_metrics: Dict[str, Any],
                                   cash_flow_metrics: Dict[str, Any],
                                   per_share_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate financial ratios from standardized metrics."""
        ratios = {}
        
        # Profitability Ratios
        
        # Return on Equity (ROE)
        if "net_income" in income_metrics and "shareholders_equity" in balance_metrics and balance_metrics["shareholders_equity"] > 0:
            ratios["roe"] = (income_metrics["net_income"] / balance_metrics["shareholders_equity"]) * 100
        
        # Return on Assets (ROA)
        if "net_income" in income_metrics and "total_assets" in balance_metrics and balance_metrics["total_assets"] > 0:
            ratios["roa"] = (income_metrics["net_income"] / balance_metrics["total_assets"]) * 100
        
        # Return on Capital Employed (ROCE)
        if ("operating_income" in income_metrics and "total_assets" in balance_metrics and 
            "current_liabilities" in balance_metrics and 
            (balance_metrics["total_assets"] - balance_metrics["current_liabilities"]) > 0):
            
            capital_employed = balance_metrics["total_assets"] - balance_metrics["current_liabilities"]
            ratios["roce"] = (income_metrics["operating_income"] / capital_employed) * 100
        
        # Gross Margin
        if "gross_profit" in income_metrics and "revenue" in income_metrics and income_metrics["revenue"] > 0:
            ratios["gross_margin"] = (income_metrics["gross_profit"] / income_metrics["revenue"]) * 100
        
        # Operating Margin
        if "operating_income" in income_metrics and "revenue" in income_metrics and income_metrics["revenue"] > 0:
            ratios["operating_margin"] = (income_metrics["operating_income"] / income_metrics["revenue"]) * 100
        
        # Net Profit Margin
        if "net_income" in income_metrics and "revenue" in income_metrics and income_metrics["revenue"] > 0:
            ratios["net_margin"] = (income_metrics["net_income"] / income_metrics["revenue"]) * 100
        
        # EBITDA Margin
        if "ebitda" in income_metrics and "revenue" in income_metrics and income_metrics["revenue"] > 0:
            ratios["ebitda_margin"] = (income_metrics["ebitda"] / income_metrics["revenue"]) * 100
        
        # Liquidity Ratios
        
        # Current Ratio
        if "current_assets" in balance_metrics and "current_liabilities" in balance_metrics and balance_metrics["current_liabilities"] > 0:
            ratios["current_ratio"] = balance_metrics["current_assets"] / balance_metrics["current_liabilities"]
        
        # Quick Ratio
        if ("current_assets" in balance_metrics and "inventory" in balance_metrics and 
            "current_liabilities" in balance_metrics and balance_metrics["current_liabilities"] > 0):
            
            quick_assets = balance_metrics["current_assets"] - balance_metrics["inventory"]
            ratios["quick_ratio"] = quick_assets / balance_metrics["current_liabilities"]
        
        # Cash Ratio
        if "cash_and_equivalents" in balance_metrics and "current_liabilities" in balance_metrics and balance_metrics["current_liabilities"] > 0:
            ratios["cash_ratio"] = balance_metrics["cash_and_equivalents"] / balance_metrics["current_liabilities"]
        
        # Leverage Ratios
        
        # Debt-to-Equity Ratio
        if "total_debt" in balance_metrics and "shareholders_equity" in balance_metrics and balance_metrics["shareholders_equity"] > 0:
            ratios["debt_to_equity"] = balance_metrics["total_debt"] / balance_metrics["shareholders_equity"]
        
        # Debt-to-Assets Ratio
        if "total_debt" in balance_metrics and "total_assets" in balance_metrics and balance_metrics["total_assets"] > 0:
            ratios["debt_to_assets"] = balance_metrics["total_debt"] / balance_metrics["total_assets"]
        
        # Interest Coverage Ratio
        if "operating_income" in income_metrics and "interest_expense" in income_metrics and income_metrics["interest_expense"] != 0:
            ratios["interest_coverage"] = income_metrics["operating_income"] / abs(income_metrics["interest_expense"])
        
        # Debt-to-EBITDA Ratio
        if "total_debt" in balance_metrics and "ebitda" in income_metrics and income_metrics["ebitda"] > 0:
            ratios["debt_to_ebitda"] = balance_metrics["total_debt"] / income_metrics["ebitda"]
        
        # Efficiency Ratios
        
        # Asset Turnover Ratio
        if "revenue" in income_metrics and "total_assets" in balance_metrics and balance_metrics["total_assets"] > 0:
            ratios["asset_turnover"] = income_metrics["revenue"] / balance_metrics["total_assets"]
        
        # Inventory Turnover Ratio
        if "cost_of_revenue" in income_metrics and "inventory" in balance_metrics and balance_metrics["inventory"] > 0:
            ratios["inventory_turnover"] = income_metrics["cost_of_revenue"] / balance_metrics["inventory"]
        
        # Receivables Turnover Ratio
        if "revenue" in income_metrics and "accounts_receivable" in balance_metrics and balance_metrics["accounts_receivable"] > 0:
            ratios["receivables_turnover"] = income_metrics["revenue"] / balance_metrics["accounts_receivable"]
        
        # Cash Flow Ratios
        
        # Operating Cash Flow to Sales Ratio
        if "operating_cash_flow" in cash_flow_metrics and "revenue" in income_metrics and income_metrics["revenue"] > 0:
            ratios["ocf_to_sales"] = (cash_flow_metrics["operating_cash_flow"] / income_metrics["revenue"]) * 100
        
        # Free Cash Flow to Operating Cash Flow Ratio
        if "free_cash_flow" in cash_flow_metrics and "operating_cash_flow" in cash_flow_metrics and cash_flow_metrics["operating_cash_flow"] > 0:
            ratios["fcf_to_ocf"] = (cash_flow_metrics["free_cash_flow"] / cash_flow_metrics["operating_cash_flow"]) * 100
        
        # Cash Flow to Net Income Ratio (Quality of Earnings)
        if "operating_cash_flow" in cash_flow_metrics and "net_income" in income_metrics and income_metrics["net_income"] > 0:
            ratios["cash_flow_to_income"] = cash_flow_metrics["operating_cash_flow"] / income_metrics["net_income"]
        
        # Capital Expenditure to Revenue Ratio
        if "capital_expenditure" in cash_flow_metrics and "revenue" in income_metrics and income_metrics["revenue"] > 0:
            # Capital expenditure is usually negative (cash outflow)
            capex = abs(cash_flow_metrics["capital_expenditure"])
            ratios["capex_to_revenue"] = (capex / income_metrics["revenue"]) * 100
        
        # Dividend Ratios
        
        # Dividend Payout Ratio
        if "dividend_per_share" in per_share_metrics and "eps" in per_share_metrics and per_share_metrics["eps"] > 0:
            ratios["payout_ratio"] = (per_share_metrics["dividend_per_share"] / per_share_metrics["eps"]) * 100
        
        # Dividend Coverage Ratio
        if "eps" in per_share_metrics and "dividend_per_share" in per_share_metrics and per_share_metrics["dividend_per_share"] > 0:
            ratios["dividend_coverage"] = per_share_metrics["eps"] / per_share_metrics["dividend_per_share"]
        
        return ratios
    
    def _get_current_market_price(self, symbol: str, exchange: str) -> float:
        """Get the current market price of a stock."""
        try:
            # Try to get the latest market data
            market_data = self.db.market_data_collection.find_one(
                {
                    "symbol": symbol, 
                    "exchange": exchange,
                    "timeframe": "day"  # Daily timeframe for closing price
                },
                sort=[("timestamp", -1)]  # Most recent first
            )
            
            if market_data and "close" in market_data:
                return market_data["close"]
            
            # If not available in market data, try financial data
            financial_data = self.db.financial_data_collection.find_one(
                {"symbol": symbol, "exchange": exchange},
                sort=[("timestamp", -1)]
            )
            
            if financial_data and "current_price" in financial_data:
                return financial_data["current_price"]
            
            self.logger.warning(f"No current market price found for {symbol}:{exchange}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error getting current market price: {e}")
            return 0
    
    def _calculate_valuation_metrics(self, per_share_metrics: Dict[str, Any], 
                                    market_price: float,
                                    financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate valuation metrics using market price."""
        metrics = {}
        
        if market_price <= 0:
            self.logger.warning("Market price not available for valuation metrics")
            return metrics
        
        # Price-to-Earnings (P/E) Ratio
        if "eps" in per_share_metrics and per_share_metrics["eps"] > 0:
            metrics["pe_ratio"] = market_price / per_share_metrics["eps"]
        
        # Price-to-Book (P/B) Ratio
        if "book_value_per_share" in per_share_metrics and per_share_metrics["book_value_per_share"] > 0:
            metrics["pb_ratio"] = market_price / per_share_metrics["book_value_per_share"]
        
        # Price-to-Sales (P/S) Ratio
        if "revenue_per_share" in per_share_metrics and per_share_metrics["revenue_per_share"] > 0:
            metrics["ps_ratio"] = market_price / per_share_metrics["revenue_per_share"]
        
        # Price-to-Cash-Flow (P/CF) Ratio
        if "ocf_per_share" in per_share_metrics and per_share_metrics["ocf_per_share"] > 0:
            metrics["pcf_ratio"] = market_price / per_share_metrics["ocf_per_share"]
        
        # Price-to-Free-Cash-Flow (P/FCF) Ratio
        if "fcf_per_share" in per_share_metrics and per_share_metrics["fcf_per_share"] > 0:
            metrics["pfcf_ratio"] = market_price / per_share_metrics["fcf_per_share"]
        
        # Dividend Yield
        if "dividend_per_share" in per_share_metrics and per_share_metrics["dividend_per_share"] > 0:
            metrics["dividend_yield"] = (per_share_metrics["dividend_per_share"] / market_price) * 100
        
        # Enterprise Value (EV)
        market_cap = 0
        if "shares_outstanding" in financial_data:
            market_cap = market_price * financial_data["shares_outstanding"]
            metrics["market_cap"] = market_cap
        
        # If we have market cap and debt/cash information, calculate EV
        if market_cap > 0:
            enterprise_value = market_cap
            
            # Add debt
            if "total_debt" in financial_data.get("balance_sheet", {}):
                enterprise_value += financial_data["balance_sheet"]["total_debt"]
            
            # Subtract cash
            if "cash_and_equivalents" in financial_data.get("balance_sheet", {}):
                enterprise_value -= financial_data["balance_sheet"]["cash_and_equivalents"]
            
            metrics["enterprise_value"] = enterprise_value
            
            # EV/EBITDA
            if "ebitda" in financial_data.get("income_statement", {}) and financial_data["income_statement"]["ebitda"] > 0:
                metrics["ev_ebitda"] = enterprise_value / financial_data["income_statement"]["ebitda"]
            
            # EV/Sales (Revenue)
            if "revenue" in financial_data.get("income_statement", {}) and financial_data["income_statement"]["revenue"] > 0:
                metrics["ev_sales"] = enterprise_value / financial_data["income_statement"]["revenue"]
            
            # EV/FCF
            if "free_cash_flow" in financial_data.get("cash_flow", {}) and financial_data["cash_flow"]["free_cash_flow"] > 0:
                metrics["ev_fcf"] = enterprise_value / financial_data["cash_flow"]["free_cash_flow"]
        
        # PEG Ratio (P/E to Growth)
        if "pe_ratio" in metrics and "eps_growth" in financial_data.get("growth_metrics", {}) and financial_data["growth_metrics"]["eps_growth"] > 0:
            metrics["peg_ratio"] = metrics["pe_ratio"] / financial_data["growth_metrics"]["eps_growth"]
        
        # Earnings Yield (inverse of P/E)
        if "pe_ratio" in metrics and metrics["pe_ratio"] > 0:
            metrics["earnings_yield"] = (1 / metrics["pe_ratio"]) * 100
        
        # FCF Yield
        if "fcf_per_share" in per_share_metrics and per_share_metrics["fcf_per_share"] > 0:
            metrics["fcf_yield"] = (per_share_metrics["fcf_per_share"] / market_price) * 100
        
        # Graham Number (Intrinsic value estimate)
        if "eps" in per_share_metrics and "book_value_per_share" in per_share_metrics:
            eps = per_share_metrics["eps"]
            bvps = per_share_metrics["book_value_per_share"]
            
            if eps > 0 and bvps > 0:
                metrics["graham_number"] = math.sqrt(15 * 1.5 * eps * bvps)
                metrics["percent_to_graham"] = ((market_price - metrics["graham_number"]) / metrics["graham_number"]) * 100
        
        return metrics
    
    def _calculate_growth_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate growth metrics from financial data."""
        metrics = {}
        
        # Extract growth metrics if already calculated by the scraper
        if "growth_metrics" in financial_data:
            return financial_data["growth_metrics"]
        
        # Check for quarterly data
        if "quarterly_results" in financial_data:
            quarterly = financial_data["quarterly_results"]
            
            # If we have at least 5 quarters for YoY comparison
            if len(quarterly) >= 5:
                # Revenue YoY Growth
                if "sales" in quarterly[0] and "sales" in quarterly[4]:
                    current = quarterly[0]["sales"]
                    prev_year = quarterly[4]["sales"]
                    if prev_year > 0:
                        metrics["revenue_growth_yoy"] = ((current - prev_year) / prev_year) * 100
                
                # Net Income YoY Growth
                if "net_profit" in quarterly[0] and "net_profit" in quarterly[4]:
                    current = quarterly[0]["net_profit"]
                    prev_year = quarterly[4]["net_profit"]
                    if prev_year > 0:
                        metrics["net_income_growth_yoy"] = ((current - prev_year) / prev_year) * 100
            
            # If we have at least 2 quarters for QoQ comparison
            if len(quarterly) >= 2:
                # Revenue QoQ Growth
                if "sales" in quarterly[0] and "sales" in quarterly[1]:
                    current = quarterly[0]["sales"]
                    prev_quarter = quarterly[1]["sales"]
                    if prev_quarter > 0:
                        metrics["revenue_growth_qoq"] = ((current - prev_quarter) / prev_quarter) * 100
                
                # Net Income QoQ Growth
                if "net_profit" in quarterly[0] and "net_profit" in quarterly[1]:
                    current = quarterly[0]["net_profit"]
                    prev_quarter = quarterly[1]["net_profit"]
                    if prev_quarter > 0:
                        metrics["net_income_growth_qoq"] = ((current - prev_quarter) / prev_quarter) * 100
        
        # Check for annual historical data
        if "historical_financials" in financial_data:
            history = financial_data["historical_financials"]
            
            # Sort by date if needed (most recent first)
            history = sorted(history, key=lambda x: x.get("date", ""), reverse=True)
            
            # If we have at least 2 years for annual growth
            if len(history) >= 2:
                # Revenue Growth
                if "revenue" in history[0] and "revenue" in history[1]:
                    current = history[0]["revenue"]
                    prev_year = history[1]["revenue"]
                    if prev_year > 0:
                        metrics["revenue_growth"] = ((current - prev_year) / prev_year) * 100
                
                # EPS Growth
                if "eps" in history[0] and "eps" in history[1]:
                    current = history[0]["eps"]
                    prev_year = history[1]["eps"]
                    if prev_year > 0:
                        metrics["eps_growth"] = ((current - prev_year) / prev_year) * 100
                
                # Net Income Growth
                if "net_income" in history[0] and "net_income" in history[1]:
                    current = history[0]["net_income"]
                    prev_year = history[1]["net_income"]
                    if prev_year > 0:
                        metrics["net_income_growth"] = ((current - prev_year) / prev_year) * 100
                
                # Book Value Growth
                if "book_value_per_share" in history[0] and "book_value_per_share" in history[1]:
                    current = history[0]["book_value_per_share"]
                    prev_year = history[1]["book_value_per_share"]
                    if prev_year > 0:
                        metrics["book_value_growth"] = ((current - prev_year) / prev_year) * 100
                
                # FCF Growth
                if "free_cash_flow" in history[0] and "free_cash_flow" in history[1]:
                    current = history[0]["free_cash_flow"]
                    prev_year = history[1]["free_cash_flow"]
                    if prev_year > 0:
                        metrics["fcf_growth"] = ((current - prev_year) / prev_year) * 100
            
            # If we have at least 5 years for CAGR calculation
            if len(history) >= 5:
                # Calculate 5-year CAGR for key metrics
                years = 5
                
                # Revenue CAGR
                if "revenue" in history[0] and "revenue" in history[min(4, len(history)-1)]:
                    start_value = history[min(4, len(history)-1)]["revenue"]
                    end_value = history[0]["revenue"]
                    if start_value > 0:
                        cagr = (pow(end_value / start_value, 1/years) - 1) * 100
                        metrics["revenue_cagr_5yr"] = cagr
                
                # EPS CAGR
                if "eps" in history[0] and "eps" in history[min(4, len(history)-1)]:
                    start_value = history[min(4, len(history)-1)]["eps"]
                    end_value = history[0]["eps"]
                    if start_value > 0:
                        cagr = (pow(end_value / start_value, 1/years) - 1) * 100
                        metrics["eps_cagr_5yr"] = cagr
                
                # Book Value CAGR
                if "book_value_per_share" in history[0] and "book_value_per_share" in history[min(4, len(history)-1)]:
                    start_value = history[min(4, len(history)-1)]["book_value_per_share"]
                    end_value = history[0]["book_value_per_share"]
                    if start_value > 0:
                        cagr = (pow(end_value / start_value, 1/years) - 1) * 100
                        metrics["book_value_cagr_5yr"] = cagr
        
        # Dividend Growth
        if "dividend_history" in financial_data:
            dividend_history = financial_data["dividend_history"]
            
            # Sort by date if needed (most recent first)
            dividend_history = sorted(dividend_history, key=lambda x: x.get("date", ""), reverse=True)
            
            # If we have at least 2 years for annual growth
            if len(dividend_history) >= 2:
                # Get most recent and previous year dividends
                current_div = dividend_history[0].get("amount", 0)
                prev_div = dividend_history[1].get("amount", 0)
                
                if prev_div > 0:
                    metrics["dividend_growth"] = ((current_div - prev_div) / prev_div) * 100
            
            # If we have at least 5 years for CAGR calculation
            if len(dividend_history) >= 5:
                # Calculate 5-year dividend CAGR
                years = 5
                start_value = dividend_history[min(4, len(dividend_history)-1)].get("amount", 0)
                end_value = dividend_history[0].get("amount", 0)
                
                if start_value > 0:
                    cagr = (pow(end_value / start_value, 1/years) - 1) * 100
                    metrics["dividend_cagr_5yr"] = cagr
            
            # Track dividend consistency
            metrics["dividend_years"] = len(dividend_history)
            
            # Count consecutive increases
            increases = 0
            for i in range(len(dividend_history) - 1):
                if dividend_history[i].get("amount", 0) > dividend_history[i+1].get("amount", 0):
                    increases += 1
                else:
                    break
            
            metrics["consecutive_dividend_increases"] = increases
        
        return metrics
    
    def _save_to_database(self, metrics: Dict[str, Any]) -> bool:
        """Save processed metrics to the database."""
        try:
            # Check if we need to update existing metrics
            existing = self.db.key_metrics_collection.find_one({
                "symbol": metrics["symbol"],
                "exchange": metrics["exchange"]
            }, sort=[("timestamp", -1)])
            
            # Get today's date
            today = datetime.now().date()
            
            if existing:
                # Update if same-day metrics, otherwise insert new
                existing_date = existing["timestamp"].date() if "timestamp" in existing else None
                
                if existing_date and existing_date == today:
                    # Update existing document
                    self.db.key_metrics_collection.update_one(
                        {"_id": existing["_id"]},
                        {"$set": metrics}
                    )
                    self.logger.info(f"Updated existing metrics for {metrics['symbol']}:{metrics['exchange']}")
                else:
                    # Insert new document
                    self.db.key_metrics_collection.insert_one(metrics)
                    self.logger.info(f"Inserted new metrics for {metrics['symbol']}:{metrics['exchange']}")
            else:
                # Insert new document
                self.db.key_metrics_collection.insert_one(metrics)
                self.logger.info(f"Inserted new metrics for {metrics['symbol']}:{metrics['exchange']}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving metrics to database: {e}")
            return False
    
    def get_latest_metrics(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get the most recent fundamental metrics for a symbol.
        
        Args:
            symbol: Stock symbol/ticker
            exchange: Stock exchange
            
        Returns:
            Dictionary containing latest metrics or empty dict if not found
        """
        try:
            latest = self.db.key_metrics_collection.find_one(
                {"symbol": symbol, "exchange": exchange},
                sort=[("timestamp", -1)]
            )
            
            if latest:
                return latest
            else:
                self.logger.warning(f"No fundamental metrics found for {symbol}:{exchange}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error retrieving latest metrics: {e}")
            return {}
    
    def compare_with_sector(self, symbol: str, exchange: str, sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare a stock's fundamental metrics with sector averages.
        
        Args:
            symbol: Stock symbol/ticker
            exchange: Stock exchange
            sector: Optional sector name (if None, uses the stock's sector)
            
        Returns:
            Dictionary containing sector comparison metrics
        """
        try:
            # Get target company metrics
            target_metrics = self.get_latest_metrics(symbol, exchange)
            
            if not target_metrics:
                return {"status": "error", "message": "No metrics found for target company"}
            
            # If sector not provided, use the one from target metrics
            if not sector and "sector" in target_metrics:
                sector = target_metrics["sector"]
            
            if not sector:
                return {"status": "error", "message": "No sector specified for comparison"}
            
            # Find other stocks in the same sector
            sector_stocks = list(self.db.key_metrics_collection.find({
                "sector": sector,
                "symbol": {"$ne": symbol}  # Exclude the target company
            }, sort=[("timestamp", -1)]))
            
            if not sector_stocks:
                return {"status": "error", "message": f"No other stocks found in sector: {sector}"}
            
            # Get unique symbols (most recent metrics for each)
            unique_stocks = {}
            for stock in sector_stocks:
                sym = stock["symbol"]
                if sym not in unique_stocks:
                    unique_stocks[sym] = stock
            
            stocks_list = list(unique_stocks.values())
            
            # Calculate sector averages
            sector_avg = {
                "pe_ratio": self._calc_avg(stocks_list, ["valuation", "pe_ratio"]),
                "pb_ratio": self._calc_avg(stocks_list, ["valuation", "pb_ratio"]),
                "ps_ratio": self._calc_avg(stocks_list, ["valuation", "ps_ratio"]),
                "dividend_yield": self._calc_avg(stocks_list, ["valuation", "dividend_yield"]),
                "roe": self._calc_avg(stocks_list, ["ratios", "roe"]),
                "roa": self._calc_avg(stocks_list, ["ratios", "roa"]),
                "debt_to_equity": self._calc_avg(stocks_list, ["ratios", "debt_to_equity"]),
                "operating_margin": self._calc_avg(stocks_list, ["ratios", "operating_margin"]),
                "net_margin": self._calc_avg(stocks_list, ["ratios", "net_margin"]),
                "revenue_growth": self._calc_avg(stocks_list, ["growth", "revenue_growth"]),
                "eps_growth": self._calc_avg(stocks_list, ["growth", "eps_growth"])
            }
            
            # Calculate percentile rankings
            percentiles = {}
            
            # Function to calculate percentile rank
            def calc_percentile(target_val, values, higher_is_better=True):
                if not values or target_val is None:
                    return None
                    
                count = 0
                for val in values:
                    if higher_is_better and val <= target_val:
                        count += 1
                    elif not higher_is_better and val >= target_val:
                        count += 1
                
                return (count / len(values)) * 100
            
            # Extract target company values
            target_vals = {
                "pe_ratio": self._extract_nested(target_metrics, ["valuation", "pe_ratio"]),
                "pb_ratio": self._extract_nested(target_metrics, ["valuation", "pb_ratio"]),
                "ps_ratio": self._extract_nested(target_metrics, ["valuation", "ps_ratio"]),
                "dividend_yield": self._extract_nested(target_metrics, ["valuation", "dividend_yield"]),
                "roe": self._extract_nested(target_metrics, ["ratios", "roe"]),
                "roa": self._extract_nested(target_metrics, ["ratios", "roa"]),
                "debt_to_equity": self._extract_nested(target_metrics, ["ratios", "debt_to_equity"]),
                "operating_margin": self._extract_nested(target_metrics, ["ratios", "operating_margin"]),
                "net_margin": self._extract_nested(target_metrics, ["ratios", "net_margin"]),
                "revenue_growth": self._extract_nested(target_metrics, ["growth", "revenue_growth"]),
                "eps_growth": self._extract_nested(target_metrics, ["growth", "eps_growth"])
            }
            
            # Calculate percentiles
            # For these metrics, lower is better (percentile shows % of stocks with higher values)
            for metric in ["pe_ratio", "pb_ratio", "ps_ratio", "debt_to_equity"]:
                if target_vals[metric] is not None:
                    values = [self._extract_nested(s, metric.split("_")) for s in stocks_list]
                    values = [v for v in values if v is not None]
                    percentiles[metric] = calc_percentile(target_vals[metric], values, False)
            
            # For these metrics, higher is better (percentile shows % of stocks with lower values)
            for metric in ["dividend_yield", "roe", "roa", "operating_margin", "net_margin", 
                          "revenue_growth", "eps_growth"]:
                if target_vals[metric] is not None:
                    values = [self._extract_nested(s, metric.split("_")) for s in stocks_list]
                    values = [v for v in values if v is not None]
                    percentiles[metric] = calc_percentile(target_vals[metric], values, True)
            
            # Build comparison result
            comparison = {
                "symbol": symbol,
                "exchange": exchange,
                "sector": sector,
                "peer_count": len(stocks_list),
                "sector_averages": sector_avg,
                "target_values": target_vals,
                "percentile_rankings": percentiles,
                "comparison_date": datetime.now()
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error performing sector comparison: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calc_avg(self, stocks_list: List[Dict[str, Any]], path: List[str]) -> Optional[float]:
        """Calculate average value from a list of stocks for a specific metric path."""
        values = []
        
        for stock in stocks_list:
            val = self._extract_nested(stock, path)
            if val is not None:
                values.append(val)
        
        if values:
            return sum(values) / len(values)
        
        return None
    
    def _extract_nested(self, data: Dict[str, Any], path: List[str]) -> Optional[Any]:
        """Extract a value from nested dictionaries following the specified path."""
        current = data
        
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


# Usage example
if __name__ == "__main__":
    # This would be used for testing only
    from pymongo import MongoClient
    
    # Example connection to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["automated_trading"]
    
    # Initialize calculator
    metrics_calculator = FundamentalMetrics(db)
    
    # Example usage - process financial data
    financial_data = {
        # Sample financial data would go here
        "income_statement": {
            "revenue": 10000000000,
            "cost_of_revenue": 6000000000,
            "gross_profit": 4000000000,
            "operating_expenses": 2000000000,
            "operating_income": 2000000000,
            "interest_expense": 200000000,
            "income_before_tax": 1800000000,
            "income_tax_expense": 360000000,
            "net_income": 1440000000,
            "ebitda": 2500000000
        },
        "balance_sheet": {
            "current_assets": 5000000000,
            "cash_and_equivalents": 1500000000,
            "inventory": 1000000000,
            "accounts_receivable": 800000000,
            "non_current_assets": 10000000000,
            "ppe": 7000000000,
            "total_assets": 15000000000,
            "current_liabilities": 3000000000,
            "accounts_payable": 1000000000,
            "short_term_debt": 500000000,
            "non_current_liabilities": 6000000000,
            "long_term_debt": 5000000000,
            "total_liabilities": 9000000000,
            "shareholders_equity": 6000000000
        },
        "cash_flow": {
            "operating_cash_flow": 2000000000,
            "capital_expenditure": -800000000,
            "free_cash_flow": 1200000000,
            "dividends_paid": -500000000
        },
        "shares_outstanding": 1000000000,
        "sector": "Technology",
        "industry": "Software"
    }
    
    # Process the financial data
    results = metrics_calculator.process_raw_financial_data(financial_data, "EXAMPLE", "NYSE")
    print(results)