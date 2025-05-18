# fundamental_features.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FundamentalFeatureGenerator:
    def __init__(self, db_connector):
        """Initialize the fundamental feature generator"""
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        
        # Initialize scalers
        self.ratio_scaler = StandardScaler()
        self.growth_scaler = StandardScaler()
        self.market_cap_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.value_scaler = StandardScaler()
    
    def generate_features(self, symbol, exchange, include_sector=True, 
                        include_historical=True, for_date=None):
        """
        Generate fundamental features for a symbol
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - include_sector: Whether to include sector-relative features
        - include_historical: Whether to include historical trends
        - for_date: Optional specific date to generate features for

        Returns:
        - Dictionary with fundamental features
        """
        try:
            # Get current financial data
            financial_data = self._get_financial_data(symbol, exchange, for_date)
            if not financial_data:
                self.logger.warning(f"No financial data found for {symbol}")
                return None
            
            # Initialize features dictionary
            features = {"symbol": symbol, "exchange": exchange}
            
            # Add basic financial metrics
            self._add_basic_metrics(features, financial_data)
            
            # Add profitability features
            self._add_profitability_features(features, financial_data)
            
            # Add valuation features
            self._add_valuation_features(features, financial_data)
            
            # Add growth features
            self._add_growth_features(features, financial_data)
            
            # Add financial health features
            self._add_financial_health_features(features, financial_data)
            
            # Add efficiency features
            self._add_efficiency_features(features, financial_data)
            
            # Add dividend features
            self._add_dividend_features(features, financial_data)
            
            # Add cash flow features
            self._add_cash_flow_features(features, financial_data)
            
            # Add sector relative features if requested
            if include_sector:
                self._add_sector_relative_features(features, symbol, exchange, financial_data)
            
            # Add historical trends if requested
            if include_historical:
                self._add_historical_trends(features, symbol, exchange, financial_data)
            
            # Add macro context
            self._add_macro_context(features, symbol, exchange, for_date)
            
            # Prefix all features with 'feature_'
            prefixed_features = {}
            
            for key, value in features.items():
                if key in ['symbol', 'exchange', 'sector', 'industry']:
                    prefixed_features[key] = value
                else:
                    prefixed_features[f"feature_{key}"] = value
            
            return prefixed_features
            
        except Exception as e:
            self.logger.error(f"Error generating fundamental features: {str(e)}")
            return None
    
    def _get_financial_data(self, symbol, exchange, for_date=None):
        """Get latest financial data for a symbol"""
        try:
            # Base query for financial data
            query = {
                "symbol": symbol,
                "exchange": exchange
            }
            
            # Add date filter if specified
            if for_date:
                query["report_date"] = {"$lte": for_date}
            
            # Get the latest financial data
            financial_data = self.db.financial_data_collection.find_one(
                query,
                sort=[("report_date", -1)]
            )
            
            return financial_data
            
        except Exception as e:
            self.logger.error(f"Error getting financial data: {str(e)}")
            return None
    
    def _add_basic_metrics(self, features, financial_data):
        """Add basic financial metrics to features"""
        try:
            # Add basic company metrics
            features["market_cap"] = financial_data.get("market_cap")
            features["enterprise_value"] = financial_data.get("enterprise_value")
            features["revenue"] = financial_data.get("revenue")
            features["ebitda"] = financial_data.get("ebitda")
            features["net_income"] = financial_data.get("net_income")
            features["eps"] = financial_data.get("eps")
            features["book_value_per_share"] = financial_data.get("book_value_per_share")
            features["shares_outstanding"] = financial_data.get("shares_outstanding")
            features["sector"] = financial_data.get("sector")
            features["industry"] = financial_data.get("industry")
            
            # Market cap category (small, mid, large cap)
            if features["market_cap"]:
                if features["market_cap"] < 2000:  # 2000 Cr for small cap (₹)
                    features["market_cap_category"] = 0  # Small cap
                elif features["market_cap"] < 10000:  # 10000 Cr for mid cap (₹)
                    features["market_cap_category"] = 1  # Mid cap
                else:
                    features["market_cap_category"] = 2  # Large cap
            
        except Exception as e:
            self.logger.error(f"Error adding basic metrics: {str(e)}")
    
    def _add_profitability_features(self, features, financial_data):
        """Add profitability metrics to features"""
        try:
            # Margins
            features["gross_margin"] = financial_data.get("gross_margin")
            features["operating_margin"] = financial_data.get("operating_margin")
            features["net_margin"] = financial_data.get("net_margin")
            features["fcf_margin"] = financial_data.get("fcf_margin")
            
            # Return metrics
            features["roe"] = financial_data.get("roe")  # Return on Equity
            features["roa"] = financial_data.get("roa")  # Return on Assets
            features["roic"] = financial_data.get("roic")  # Return on Invested Capital
            
            # Margin stability (if available from historical data)
            if "margin_stability" in financial_data:
                features["margin_stability"] = financial_data.get("margin_stability")
            
            # Earnings quality
            features["accrual_ratio"] = financial_data.get("accrual_ratio")
            
            # Revenue per employee (efficiency)
            if financial_data.get("revenue") and financial_data.get("employees"):
                features["revenue_per_employee"] = financial_data.get("revenue") / financial_data.get("employees")
            
            # Net income per employee
            if financial_data.get("net_income") and financial_data.get("employees"):
                features["income_per_employee"] = financial_data.get("net_income") / financial_data.get("employees")
            
            # Tax rate
            features["effective_tax_rate"] = financial_data.get("effective_tax_rate")
            
        except Exception as e:
            self.logger.error(f"Error adding profitability features: {str(e)}")
    
    def _add_valuation_features(self, features, financial_data):
        """Add valuation metrics to features"""
        try:
            # Price multiples
            features["pe_ratio"] = financial_data.get("pe_ratio")  # Price-to-Earnings
            features["forward_pe"] = financial_data.get("forward_pe")  # Forward P/E
            features["pb_ratio"] = financial_data.get("pb_ratio")  # Price-to-Book
            features["ps_ratio"] = financial_data.get("ps_ratio")  # Price-to-Sales
            features["pfcf_ratio"] = financial_data.get("pfcf_ratio")  # Price-to-FCF
            
            # Enterprise value multiples
            features["ev_to_ebitda"] = financial_data.get("ev_to_ebitda")
            features["ev_to_sales"] = financial_data.get("ev_to_sales")
            features["ev_to_fcf"] = financial_data.get("ev_to_fcf")
            
            # PEG ratio (P/E to Growth)
            features["peg_ratio"] = financial_data.get("peg_ratio")
            
            # Earnings yield (inverse of P/E)
            if features["pe_ratio"] and features["pe_ratio"] > 0:
                features["earnings_yield"] = 1 / features["pe_ratio"]
            
            # FCF yield
            features["fcf_yield"] = financial_data.get("fcf_yield")
            
            # Intrinsic value estimates (if available)
            features["graham_number"] = financial_data.get("graham_number")
            features["dcf_value"] = financial_data.get("dcf_value")
            
            # Value gap (price to intrinsic value ratio)
            if financial_data.get("current_price") and financial_data.get("dcf_value"):
                features["value_gap"] = financial_data.get("current_price") / financial_data.get("dcf_value") - 1
            
            # Tobin's Q (approximation)
            if financial_data.get("market_cap") and financial_data.get("total_assets") and financial_data.get("total_liabilities"):
                book_value = financial_data.get("total_assets") - financial_data.get("total_liabilities")
                if book_value > 0:
                    features["tobins_q"] = financial_data.get("market_cap") / book_value
            
        except Exception as e:
            self.logger.error(f"Error adding valuation features: {str(e)}")
    
    def _add_growth_features(self, features, financial_data):
        """Add growth metrics to features"""
        try:
            # Year-over-Year growth rates
            features["revenue_growth"] = financial_data.get("revenue_growth")
            features["earnings_growth"] = financial_data.get("earnings_growth")
            features["fcf_growth"] = financial_data.get("fcf_growth")
            features["book_value_growth"] = financial_data.get("book_value_growth")
            
            # CAGR (Compound Annual Growth Rate) for longer periods
            features["revenue_cagr_3y"] = financial_data.get("revenue_cagr_3y")
            features["earnings_cagr_3y"] = financial_data.get("earnings_cagr_3y")
            features["fcf_cagr_3y"] = financial_data.get("fcf_cagr_3y")
            
            # Growth stability
            features["revenue_growth_stability"] = financial_data.get("revenue_growth_stability")
            features["earnings_growth_stability"] = financial_data.get("earnings_growth_stability")
            
            # Historic P/E expansion/contraction
            features["pe_expansion_3y"] = financial_data.get("pe_expansion_3y")
            
            # Growth vs P/E ratio
            if features["pe_ratio"] and features["earnings_growth"]:
                features["pe_to_growth_ratio"] = features["pe_ratio"] / features["earnings_growth"]
            
            # Calculate reinvestment rate
            if financial_data.get("capex") and financial_data.get("net_income") and financial_data.get("net_income") != 0:
                features["reinvestment_rate"] = financial_data.get("capex") / financial_data.get("net_income")
            
            # R&D as percentage of revenue
            if financial_data.get("research_development") and financial_data.get("revenue") and financial_data.get("revenue") != 0:
                features["rd_to_revenue"] = financial_data.get("research_development") / financial_data.get("revenue")
            
        except Exception as e:
            self.logger.error(f"Error adding growth features: {str(e)}")
    
    def _add_financial_health_features(self, features, financial_data):
        """Add financial health and leverage metrics to features"""
        try:
            # Leverage ratios
            features["debt_to_equity"] = financial_data.get("debt_to_equity")
            features["debt_to_assets"] = financial_data.get("debt_to_assets")
            features["net_debt_to_ebitda"] = financial_data.get("net_debt_to_ebitda")
            
            # Liquidity ratios
            features["current_ratio"] = financial_data.get("current_ratio")
            features["quick_ratio"] = financial_data.get("quick_ratio")
            features["cash_ratio"] = financial_data.get("cash_ratio")
            
            # Interest coverage
            features["interest_coverage"] = financial_data.get("interest_coverage")
            
            # Debt service coverage
            features["debt_service_coverage"] = financial_data.get("debt_service_coverage")
            
            # Cash conversion cycle
            features["cash_conversion_cycle"] = financial_data.get("cash_conversion_cycle")
            
            # Altman Z-Score (bankruptcy risk)
            features["altman_z_score"] = financial_data.get("altman_z_score")
            
            # Piotroski F-Score (financial strength)
            features["piotroski_score"] = financial_data.get("piotroski_score")
            
            # Beneish M-Score (earnings manipulation)
            features["beneish_m_score"] = financial_data.get("beneish_m_score")
            
            # Working capital as percentage of revenue
            if financial_data.get("working_capital") and financial_data.get("revenue") and financial_data.get("revenue") != 0:
                features["working_capital_to_revenue"] = financial_data.get("working_capital") / financial_data.get("revenue")
            
            # Cash to assets ratio
            if financial_data.get("cash_and_equivalents") and financial_data.get("total_assets") and financial_data.get("total_assets") != 0:
                features["cash_to_assets"] = financial_data.get("cash_and_equivalents") / financial_data.get("total_assets")
            
            # Debt maturity profile (if available)
            features["short_term_debt_ratio"] = financial_data.get("short_term_debt_ratio")
            
        except Exception as e:
            self.logger.error(f"Error adding financial health features: {str(e)}")
    
    def _add_efficiency_features(self, features, financial_data):
        """Add operational efficiency metrics to features"""
        try:
            # Asset turnover
            features["asset_turnover"] = financial_data.get("asset_turnover")
            
            # Inventory turnover
            features["inventory_turnover"] = financial_data.get("inventory_turnover")
            
            # Receivables turnover
            features["receivables_turnover"] = financial_data.get("receivables_turnover")
            
            # Days sales outstanding (DSO)
            features["days_sales_outstanding"] = financial_data.get("days_sales_outstanding")
            
            # Days inventory outstanding (DIO)
            features["days_inventory_outstanding"] = financial_data.get("days_inventory_outstanding")
            
            # Days payable outstanding (DPO)
            features["days_payable_outstanding"] = financial_data.get("days_payable_outstanding")
            
            # Fixed asset turnover
            features["fixed_asset_turnover"] = financial_data.get("fixed_asset_turnover")
            
            # Operating cycle
            if features["days_inventory_outstanding"] and features["days_sales_outstanding"]:
                features["operating_cycle"] = features["days_inventory_outstanding"] + features["days_sales_outstanding"]
            
            # SG&A as percentage of revenue
            if financial_data.get("sg_and_a") and financial_data.get("revenue") and financial_data.get("revenue") != 0:
                features["sga_to_revenue"] = financial_data.get("sg_and_a") / financial_data.get("revenue")
            
            # Capital expenditure efficiency
            if financial_data.get("capex") and financial_data.get("depreciation") and financial_data.get("depreciation") != 0:
                features["capex_to_depreciation"] = abs(financial_data.get("capex")) / financial_data.get("depreciation")
            
            # Operating leverage
            if financial_data.get("operating_income_growth") and financial_data.get("revenue_growth"):
                features["operating_leverage"] = financial_data.get("operating_income_growth") / financial_data.get("revenue_growth")
            
        except Exception as e:
            self.logger.error(f"Error adding efficiency features: {str(e)}")
    
    def _add_dividend_features(self, features, financial_data):
        """Add dividend-related metrics to features"""
        try:
            # Dividend metrics
            features["dividend_yield"] = financial_data.get("dividend_yield")
            features["dividend_payout_ratio"] = financial_data.get("dividend_payout_ratio")
            features["dividend_per_share"] = financial_data.get("dividend_per_share")
            
            # Dividend growth
            features["dividend_growth"] = financial_data.get("dividend_growth")
            features["dividend_cagr_5y"] = financial_data.get("dividend_cagr_5y")
            
            # Consecutive dividend years
            features["dividend_years"] = financial_data.get("dividend_years")
            
            # Buyback yield
            features["buyback_yield"] = financial_data.get("buyback_yield")
            
            # Total shareholder yield (dividend + buyback)
            if features["dividend_yield"] and features["buyback_yield"]:
                features["shareholder_yield"] = features["dividend_yield"] + features["buyback_yield"]
            
            # Dividend coverage ratio
            if features["dividend_payout_ratio"] and features["dividend_payout_ratio"] != 0:
                features["dividend_coverage"] = 1 / features["dividend_payout_ratio"]
            
            # Sustainable growth rate (retention ratio * ROE)
            if features["roe"] and features["dividend_payout_ratio"]:
                features["sustainable_growth_rate"] = features["roe"] * (1 - features["dividend_payout_ratio"])
            
        except Exception as e:
            self.logger.error(f"Error adding dividend features: {str(e)}")
    
    def _add_cash_flow_features(self, features, financial_data):
        """Add cash flow metrics to features"""
        try:
            # Basic cash flow metrics
            features["fcf"] = financial_data.get("free_cash_flow")
            features["ocf"] = financial_data.get("operating_cash_flow")
            
            # Cash flow conversion ratios
            if financial_data.get("operating_cash_flow") and financial_data.get("net_income") and financial_data.get("net_income") != 0:
                features["cf_to_earnings"] = financial_data.get("operating_cash_flow") / financial_data.get("net_income")
            
            if financial_data.get("free_cash_flow") and financial_data.get("operating_cash_flow") and financial_data.get("operating_cash_flow") != 0:
                features["fcf_to_ocf"] = financial_data.get("free_cash_flow") / financial_data.get("operating_cash_flow")
            
            # Cash flow to revenue
            if financial_data.get("operating_cash_flow") and financial_data.get("revenue") and financial_data.get("revenue") != 0:
                features["cf_to_revenue"] = financial_data.get("operating_cash_flow") / financial_data.get("revenue")
            
            if financial_data.get("free_cash_flow") and financial_data.get("revenue") and financial_data.get("revenue") != 0:
                features["fcf_to_revenue"] = financial_data.get("free_cash_flow") / financial_data.get("revenue")
            
            # Cash flow to assets
            if financial_data.get("operating_cash_flow") and financial_data.get("total_assets") and financial_data.get("total_assets") != 0:
                features["cf_to_assets"] = financial_data.get("operating_cash_flow") / financial_data.get("total_assets")
            
            # Cash flow stability
            features["cf_stability"] = financial_data.get("cf_stability")
            
            # Cash conversion efficiency
            if financial_data.get("operating_cash_flow") and financial_data.get("ebitda") and financial_data.get("ebitda") != 0:
                features["cash_conversion_efficiency"] = financial_data.get("operating_cash_flow") / financial_data.get("ebitda")
            
            # Capex as percentage of operating cash flow
            if financial_data.get("capex") and financial_data.get("operating_cash_flow") and financial_data.get("operating_cash_flow") != 0:
                features["capex_to_ocf"] = abs(financial_data.get("capex")) / financial_data.get("operating_cash_flow")
            
            # Capex as percentage of revenue
            if financial_data.get("capex") and financial_data.get("revenue") and financial_data.get("revenue") != 0:
                features["capex_to_revenue"] = abs(financial_data.get("capex")) / financial_data.get("revenue")
            
            # Cash flow growth
            features["ocf_growth"] = financial_data.get("ocf_growth")
            features["fcf_growth"] = financial_data.get("fcf_growth")
            
            # Free cash flow yield
            features["fcf_yield"] = financial_data.get("fcf_yield")
            
            # Cash to debt ratio
            if financial_data.get("cash_and_equivalents") and financial_data.get("total_debt") and financial_data.get("total_debt") != 0:
                features["cash_to_debt"] = financial_data.get("cash_and_equivalents") / financial_data.get("total_debt")
            
        except Exception as e:
            self.logger.error(f"Error adding cash flow features: {str(e)}")
    
    def _add_sector_relative_features(self, features, symbol, exchange, financial_data):
        """Add features relative to sector averages"""
        try:
            sector = financial_data.get("sector")
            if not sector:
                return
            
            # Get sector average metrics
            sector_metrics = self._get_sector_metrics(sector, exchange)
            if not sector_metrics:
                return
            
            # Calculate relative metrics
            relative_metrics = [
                "pe_ratio", "pb_ratio", "ps_ratio", "ev_to_ebitda", "dividend_yield",
                "operating_margin", "net_margin", "roe", "roa", "debt_to_equity",
                "revenue_growth", "earnings_growth", "fcf_growth"
            ]
            
            for metric in relative_metrics:
                if metric in features and metric in sector_metrics and sector_metrics[metric] != 0:
                    features[f"{metric}_relative"] = features[metric] / sector_metrics[metric]
            
            # Percentile rankings within sector (if available)
            percentile_metrics = self._get_sector_percentiles(symbol, exchange, sector)
            if percentile_metrics:
                for metric, percentile in percentile_metrics.items():
                    features[f"{metric}_percentile"] = percentile
            
            # Add sector premium/discount
            if "pe_ratio_relative" in features:
                # Above 1 means trading at premium, below 1 means discount
                features["sector_valuation_gap"] = features["pe_ratio_relative"] - 1
            
            # Sector rotation indicators
            sector_momentum = self._get_sector_momentum(sector, exchange)
            if sector_momentum:
                features["sector_momentum"] = sector_momentum.get("momentum")
                features["sector_rs"] = sector_momentum.get("relative_strength")  # Relative to market
            
        except Exception as e:
            self.logger.error(f"Error adding sector relative features: {str(e)}")
    
    def _get_sector_metrics(self, sector, exchange):
        """Get average metrics for a sector"""
        try:
            # Query for sector metrics
            sector_metrics = self.db.sector_metrics_collection.find_one({
                "sector": sector,
                "exchange": exchange
            })
            
            return sector_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting sector metrics: {str(e)}")
            return None
    
    def _get_sector_percentiles(self, symbol, exchange, sector):
        """Get percentile rankings within sector"""
        try:
            # Query for percentile rankings
            percentiles = self.db.stock_percentiles_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "sector": sector
            })
            
            return percentiles
            
        except Exception as e:
            self.logger.error(f"Error getting sector percentiles: {str(e)}")
            return None
    
    def _get_sector_momentum(self, sector, exchange):
        """Get momentum metrics for a sector"""
        try:
            # Query for sector momentum
            momentum = self.db.sector_momentum_collection.find_one({
                "sector": sector,
                "exchange": exchange
            })
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error getting sector momentum: {str(e)}")
            return None
    
    def _add_historical_trends(self, features, symbol, exchange, current_data):
        """Add features about historical trends in financial metrics"""
        try:
            # Get historical financial data (last 5 years)
            historical_data = list(self.db.financial_data_collection.find(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "report_type": "annual"  # Annual reports
                }
            ).sort("report_date", -1).limit(5))
            
            if not historical_data or len(historical_data) < 3:
                return
            
            # Sort by date (oldest first)
            historical_data = sorted(historical_data, key=lambda x: x["report_date"])
            
            # Calculate trend metrics
            trend_metrics = [
                "revenue", "operating_income", "net_income", "eps",
                "operating_margin", "net_margin", "roe", "roa",
                "debt_to_equity", "current_ratio", "fcf"
            ]
            
            for metric in trend_metrics:
                values = [data.get(metric) for data in historical_data if data.get(metric) is not None]
                
                if len(values) >= 3:  # Need at least 3 points for trend
                    # Calculate simple linear regression
                    x = np.arange(len(values))
                    slope, intercept = np.polyfit(x, values, 1)
                    
                    # Normalize slope to percentage of average value
                    avg_value = np.mean(values)
                    if avg_value != 0:
                        norm_slope = slope / avg_value
                    else:
                        norm_slope = 0
                    
                    # Add trend feature
                    features[f"{metric}_trend"] = norm_slope
                    
                    # Calculate R-squared to measure trend consistency
                    y_pred = slope * x + intercept
                    ss_total = np.sum((values - avg_value) ** 2)
                    ss_residual = np.sum((values - y_pred) ** 2)
                    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                    
                    features[f"{metric}_trend_consistency"] = r_squared
            
            # Calculate earnings acceleration
            if len(historical_data) >= 4:
                eps_values = [data.get("eps") for data in historical_data if data.get("eps") is not None]
                
                if len(eps_values) >= 4:
                    # Calculate growth rates
                    growth_rates = [(eps_values[i] - eps_values[i-1]) / abs(eps_values[i-1]) if eps_values[i-1] != 0 else 0 
                                   for i in range(1, len(eps_values))]
                    
                    # Calculate acceleration (change in growth rates)
                    accelerations = [growth_rates[i] - growth_rates[i-1] for i in range(1, len(growth_rates))]
                    
                    # Average acceleration
                    features["earnings_acceleration"] = np.mean(accelerations)
            
            # Variance in metrics (stability)
            for metric in ["operating_margin", "net_margin", "roe", "fcf_to_revenue"]:
                values = [data.get(metric) for data in historical_data if data.get(metric) is not None]
                
                if len(values) >= 3:
                    features[f"{metric}_stability"] = 1 - (np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
            
            # Check for potential mean reversion signals
            for metric in ["pe_ratio", "pb_ratio", "operating_margin", "net_margin", "roe"]:
                values = [data.get(metric) for data in historical_data if data.get(metric) is not None]
                
                if len(values) >= 3 and metric in current_data:
                    current_value = current_data.get(metric)
                    
                    if current_value is not None:
                        # Calculate z-score relative to historical average
                        hist_mean = np.mean(values)
                        hist_std = np.std(values)
                        
                        if hist_std > 0:
                            z_score = (current_value - hist_mean) / hist_std
                            features[f"{metric}_z_score"] = z_score
            
        except Exception as e:
            self.logger.error(f"Error adding historical trends: {str(e)}")
    
    def _add_macro_context(self, features, symbol, exchange, for_date=None):
        """Add macroeconomic context features"""
        try:
            # Get latest macro data
            macro_data = self._get_macro_data(exchange, for_date)
            if not macro_data:
                return
            
            # Add relevant macro indicators
            features["interest_rate"] = macro_data.get("interest_rate")
            features["inflation_rate"] = macro_data.get("inflation_rate")
            features["gdp_growth"] = macro_data.get("gdp_growth")
            features["unemployment_rate"] = macro_data.get("unemployment_rate")
            features["vix_index"] = macro_data.get("vix_index")  # Market volatility
            
            # Add market trend indicators
            features["market_trend"] = macro_data.get("market_trend")  # -1 for bearish, 0 for neutral, 1 for bullish
            features["market_pe"] = macro_data.get("market_pe")  # Overall market P/E
            
            # Interest rate sensitivity
            industry = features.get("industry")
            if industry:
                rate_sensitivity = self._get_interest_rate_sensitivity(industry)
                if rate_sensitivity:
                    features["interest_rate_sensitivity"] = rate_sensitivity
            
            # International exposure
            int_exposure = self._get_international_exposure(symbol, exchange)
            if int_exposure:
                features["international_revenue_pct"] = int_exposure.get("international_revenue_pct")
                features["forex_sensitivity"] = int_exposure.get("forex_sensitivity")
            
            # Commodity price sensitivity
            comm_sensitivity = self._get_commodity_sensitivity(symbol, exchange)
            if comm_sensitivity:
                features["commodity_sensitivity"] = comm_sensitivity.get("sensitivity")
                features["key_commodity"] = comm_sensitivity.get("key_commodity")
            
            # Beta calculation (market sensitivity)
            beta = self._get_beta(symbol, exchange)
            if beta:
                features["beta"] = beta
            
        except Exception as e:
            self.logger.error(f"Error adding macro context: {str(e)}")
    
    def _get_macro_data(self, exchange, for_date=None):
        """Get macroeconomic data for a specific date"""
        try:
            # Base query for macro data
            query = {
                "exchange_country": self._get_country_from_exchange(exchange)
            }
            
            # Add date filter if specified
            if for_date:
                query["date"] = {"$lte": for_date}
            
            # Get the latest macro data
            macro_data = self.db.macro_data_collection.find_one(
                query,
                sort=[("date", -1)]
            )
            
            return macro_data
            
        except Exception as e:
            self.logger.error(f"Error getting macro data: {str(e)}")
            return None
    
    def _get_country_from_exchange(self, exchange):
        """Map exchange to country"""
        exchange_map = {
            "NSE": "India",
            "BSE": "India",
            "NYSE": "United States",
            "NASDAQ": "United States",
            "LSE": "United Kingdom",
            "TSE": "Japan"
        }
        
        return exchange_map.get(exchange, "Global")
    
    def _get_interest_rate_sensitivity(self, industry):
        """Get interest rate sensitivity for an industry"""
        try:
            # Query for industry sensitivity
            sensitivity = self.db.industry_sensitivity_collection.find_one({
                "industry": industry
            })
            
            if sensitivity:
                return sensitivity.get("interest_rate_sensitivity")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting interest rate sensitivity: {str(e)}")
            return None
    
    def _get_international_exposure(self, symbol, exchange):
        """Get international exposure data for a company"""
        try:
            # Query for international exposure
            exposure = self.db.international_exposure_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            return exposure
            
        except Exception as e:
            self.logger.error(f"Error getting international exposure: {str(e)}")
            return None
    
    def _get_commodity_sensitivity(self, symbol, exchange):
        """Get commodity price sensitivity for a company"""
        try:
            # Query for commodity sensitivity
            sensitivity = self.db.commodity_sensitivity_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            return sensitivity
            
        except Exception as e:
            self.logger.error(f"Error getting commodity sensitivity: {str(e)}")
            return None
    
    def _get_beta(self, symbol, exchange):
        """Get beta (market sensitivity) for a stock"""
        try:
            # Query for stock beta
            stock_data = self.db.stock_metadata_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            if stock_data:
                return stock_data.get("beta")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting beta: {str(e)}")
            return None
    
    def get_feature_importance(self, symbols, exchange, target='target_return_next_5d'):
        """
        Calculate feature importance across multiple stocks
        
        Parameters:
        - symbols: List of stock symbols
        - exchange: Exchange (e.g., NSE)
        - target: Target variable to predict

        Returns:
        - DataFrame with feature importances
        """
        try:
            # Generate features for all symbols
            all_features = []
            
            for symbol in symbols:
                # Get features
                features = self.generate_features(symbol, exchange)
                
                if not features:
                    continue
                
                # Get target variable from market data
                market_data = self._get_target_data(symbol, exchange, target)
                if not market_data:
                    continue
                
                # Add target to features
                features[target] = market_data
                
                all_features.append(features)
            
            if not all_features:
                return None
            
            # Convert to dataframe
            features_df = pd.DataFrame(all_features)
            
            # Remove non-feature columns
            meta_cols = ['symbol', 'exchange', 'sector', 'industry']
            feature_cols = [col for col in features_df.columns if col.startswith('feature_') and features_df[col].notna().all()]
            
            if not feature_cols or target not in features_df.columns:
                return None
            
            # Check if we have enough data
            if len(features_df) < 10:
                return None
            
            # Calculate feature importance using Random Forest
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.feature_selection import mutual_info_regression
                
                # Prepare data
                X = features_df[feature_cols].values
                y = features_df[target].values
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Get feature importance
                rf_importance = model.feature_importances_
                
                # Calculate mutual information
                mi_importance = mutual_info_regression(X, y, random_state=42)
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'random_forest_importance': rf_importance,
                    'mutual_info_importance': mi_importance,
                    'combined_importance': (rf_importance + mi_importance) / 2
                })
                
                # Sort by combined importance
                importance_df = importance_df.sort_values('combined_importance', ascending=False)
                
                return importance_df
                
            except Exception as e:
                self.logger.error(f"Error calculating feature importance: {str(e)}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error in get_feature_importance: {str(e)}")
            return None
    
    def _get_target_data(self, symbol, exchange, target):
        """Get target variable data from market data"""
        try:
            # Extract target type and horizon
            if 'return' in target:
                # Extract horizon from target string (e.g., 'target_return_next_5d')
                horizon = int(target.split('_')[-1].replace('d', ''))
                
                # Get market data
                query = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": "day"
                }
                
                # Get the latest market data
                market_data = list(self.db.market_data_collection.find(
                    query,
                    {"timestamp": 1, "close": 1}
                ).sort("timestamp", -1).limit(horizon + 1))
                
                if len(market_data) <= horizon:
                    return None
                
                # Calculate return
                current_price = market_data[-1]["close"]
                future_price = market_data[0]["close"]
                
                return (future_price / current_price) - 1
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting target data: {str(e)}")
            return None