"""
Quarterly Financial Data Analyzer

This module analyzes quarterly financial data collected by the financial scraper.
It calculates growth rates, trends, and key metrics from quarterly financial reports.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Union, Optional, Tuple, Any

class QuarterlyAnalyzer:
    """
    Analyzes quarterly financial reports to extract growth trends,
    performance metrics, and financial health indicators.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the quarterly analyzer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up a logger for this module.
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def analyze(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Analyze quarterly data for a specific symbol.
        
        Args:
            symbol: Stock symbol/ticker
            exchange: Stock exchange
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info(f"Analyzing quarterly data for {symbol}:{exchange}")
        
        try:
            # Fetch quarterly data from database
            quarterly_data = self._fetch_quarterly_data(symbol, exchange)
            
            if not quarterly_data or len(quarterly_data) < 2:
                self.logger.warning(f"Insufficient quarterly data for {symbol}:{exchange}")
                return {"status": "insufficient_data", "symbol": symbol, "exchange": exchange}
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(quarterly_data)
            
            # Sort by date (most recent first)
            df = df.sort_values('report_date', ascending=False)
            
            # Calculate analysis components
            growth_analysis = self._analyze_growth(df)
            profitability_analysis = self._analyze_profitability(df)
            consistency_analysis = self._analyze_consistency(df)
            
            # Calculate overall score
            score = self._calculate_score(growth_analysis, profitability_analysis, consistency_analysis)
            
            # Save analysis results to database
            analysis_result = {
                "symbol": symbol,
                "exchange": exchange,
                "analysis_date": datetime.now(),
                "quarters_analyzed": len(df),
                "latest_quarter": df.iloc[0]['period'],
                "growth_analysis": growth_analysis,
                "profitability_analysis": profitability_analysis,
                "consistency_analysis": consistency_analysis,
                "quarterly_score": score
            }
            
            self._save_analysis(analysis_result)
            
            self.logger.info(f"Quarterly analysis completed for {symbol}:{exchange} with score {score}")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing quarterly data for {symbol}:{exchange}: {e}")
            return {"status": "error", "symbol": symbol, "exchange": exchange, "error": str(e)}
    
    def _fetch_quarterly_data(self, symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """
        Fetch quarterly financial data from database.
        
        Args:
            symbol: Stock symbol/ticker
            exchange: Stock exchange
            
        Returns:
            List of quarterly data documents
        """
        try:
            # Query MongoDB for quarterly results
            quarterly_data = list(self.db.quarterly_results_collection.find(
                {"symbol": symbol, "exchange": exchange},
                {"_id": 0}  # Exclude MongoDB _id field
            ).sort("report_date", -1).limit(12))  # Get last 12 quarters (3 years)
            
            self.logger.info(f"Retrieved {len(quarterly_data)} quarterly records for {symbol}:{exchange}")
            return quarterly_data
            
        except Exception as e:
            self.logger.error(f"Error fetching quarterly data: {e}")
            return []
    
    def _analyze_growth(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze growth metrics from quarterly data.
        
        Args:
            df: DataFrame containing quarterly data
            
        Returns:
            Dictionary of growth metrics
        """
        # Calculate year-over-year growth rates
        try:
            # Ensure we have sorted data (newest to oldest)
            df = df.sort_values('report_date', ascending=False).reset_index(drop=True)
            
            # Initialize results
            results = {
                "revenue_growth": {},
                "profit_growth": {},
                "sequential_growth": {}
            }
            
            # Year-over-year analysis (compared to same quarter last year)
            if len(df) >= 5:  # Need at least 5 quarters for YoY analysis
                # Revenue YoY growth (last 4 quarters)
                yoy_revenue = []
                yoy_profit = []
                
                for i in range(4):
                    if i+4 < len(df) and 'sales' in df.iloc[i] and 'sales' in df.iloc[i+4]:
                        rev_growth = ((df.iloc[i]['sales'] - df.iloc[i+4]['sales']) / 
                                     df.iloc[i+4]['sales']) * 100 if df.iloc[i+4]['sales'] else 0
                        yoy_revenue.append(rev_growth)
                    
                    if i+4 < len(df) and 'net_profit' in df.iloc[i] and 'net_profit' in df.iloc[i+4]:
                        profit_growth = ((df.iloc[i]['net_profit'] - df.iloc[i+4]['net_profit']) / 
                                        df.iloc[i+4]['net_profit']) * 100 if df.iloc[i+4]['net_profit'] else 0
                        yoy_profit.append(profit_growth)
                
                results["revenue_growth"]["yoy_values"] = yoy_revenue
                results["revenue_growth"]["yoy_avg"] = np.mean(yoy_revenue) if yoy_revenue else 0
                results["revenue_growth"]["yoy_trend"] = "increasing" if len(yoy_revenue) >= 2 and yoy_revenue[0] > yoy_revenue[-1] else "decreasing"
                
                results["profit_growth"]["yoy_values"] = yoy_profit
                results["profit_growth"]["yoy_avg"] = np.mean(yoy_profit) if yoy_profit else 0
                results["profit_growth"]["yoy_trend"] = "increasing" if len(yoy_profit) >= 2 and yoy_profit[0] > yoy_profit[-1] else "decreasing"
            
            # Sequential growth (quarter-over-quarter)
            if len(df) >= 2:
                seq_revenue = []
                seq_profit = []
                
                for i in range(len(df)-1):
                    if 'sales' in df.iloc[i] and 'sales' in df.iloc[i+1]:
                        rev_growth = ((df.iloc[i]['sales'] - df.iloc[i+1]['sales']) / 
                                     df.iloc[i+1]['sales']) * 100 if df.iloc[i+1]['sales'] else 0
                        seq_revenue.append(rev_growth)
                    
                    if 'net_profit' in df.iloc[i] and 'net_profit' in df.iloc[i+1]:
                        profit_growth = ((df.iloc[i]['net_profit'] - df.iloc[i+1]['net_profit']) / 
                                        df.iloc[i+1]['net_profit']) * 100 if df.iloc[i+1]['net_profit'] else 0
                        seq_profit.append(profit_growth)
                
                results["sequential_growth"]["revenue_values"] = seq_revenue
                results["sequential_growth"]["revenue_avg"] = np.mean(seq_revenue) if seq_revenue else 0
                
                results["sequential_growth"]["profit_values"] = seq_profit
                results["sequential_growth"]["profit_avg"] = np.mean(seq_profit) if seq_profit else 0
            
            # Calculate CAGR if we have enough data
            if len(df) >= 8:  # 2 years of data
                try:
                    oldest = df.iloc[-8]['sales']
                    newest = df.iloc[0]['sales']
                    if oldest and newest:
                        # 2-year CAGR calculation
                        results["revenue_growth"]["cagr_2yr"] = (((newest / oldest) ** (1/2)) - 1) * 100
                    
                    oldest_profit = df.iloc[-8]['net_profit']
                    newest_profit = df.iloc[0]['net_profit']
                    if oldest_profit and newest_profit and oldest_profit > 0:
                        results["profit_growth"]["cagr_2yr"] = (((newest_profit / oldest_profit) ** (1/2)) - 1) * 100
                except (KeyError, ZeroDivisionError, TypeError) as e:
                    self.logger.warning(f"Error calculating CAGR: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing growth metrics: {e}")
            return {"error": str(e)}
    
    def _analyze_profitability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze profitability metrics from quarterly data.
        
        Args:
            df: DataFrame containing quarterly data
            
        Returns:
            Dictionary of profitability metrics
        """
        try:
            results = {
                "margins": {},
                "ratios": {},
                "trends": {}
            }
            
            # Calculate average margins for recent quarters (up to 4)
            recent_quarters = min(4, len(df))
            recent_df = df.head(recent_quarters)
            
            # Gross Profit Margin
            if 'sales' in recent_df.columns and 'expenses' in recent_df.columns:
                recent_df['gross_profit'] = recent_df['sales'] - recent_df['expenses']
                recent_df['gpm'] = (recent_df['gross_profit'] / recent_df['sales']) * 100
                results["margins"]["gross_margin_avg"] = recent_df['gpm'].mean()
            
            # Operating Profit Margin
            if 'operating_profit' in recent_df.columns and 'sales' in recent_df.columns:
                recent_df['opm'] = (recent_df['operating_profit'] / recent_df['sales']) * 100
                results["margins"]["operating_margin_avg"] = recent_df['opm'].mean()
            
            # Net Profit Margin
            if 'net_profit' in recent_df.columns and 'sales' in recent_df.columns:
                recent_df['npm'] = (recent_df['net_profit'] / recent_df['sales']) * 100
                results["margins"]["net_margin_avg"] = recent_df['npm'].mean()
            
            # Extract any key ratios already calculated
            if 'key_ratios' in recent_df.columns:
                # Extract common ratios if they exist
                ratio_cols = ['roce', 'roe', 'debt_to_equity']
                for ratio in ratio_cols:
                    try:
                        values = [q['key_ratios'].get(ratio, 0) for q in recent_df['key_ratios'] if q]
                        if values:
                            results["ratios"][ratio] = np.mean(values)
                    except:
                        pass
            
            # Analyze margin trends (improving or declining)
            if len(df) >= 4:
                for margin_type in ['gpm', 'opm', 'npm']:
                    if margin_type in recent_df.columns:
                        # Check trend direction
                        trend = np.polyfit(range(len(recent_df[margin_type])), recent_df[margin_type], 1)[0]
                        results["trends"][f"{margin_type}_trend"] = "improving" if trend > 0 else "declining"
                        results["trends"][f"{margin_type}_trend_value"] = float(trend)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing profitability metrics: {e}")
            return {"error": str(e)}
    
    def _analyze_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze consistency and quality of earnings.
        
        Args:
            df: DataFrame containing quarterly data
            
        Returns:
            Dictionary of consistency metrics
        """
        try:
            results = {
                "earnings_quality": {},
                "revenue_consistency": {},
                "profit_consistency": {}
            }
            
            # Check for minimum data
            if len(df) < 4:
                return {"insufficient_data": True}
            
            # Revenue consistency (coefficient of variation)
            if 'sales' in df.columns:
                rev_std = df['sales'].std()
                rev_mean = df['sales'].mean()
                if rev_mean > 0:
                    results["revenue_consistency"]["cv"] = (rev_std / rev_mean) * 100
                    results["revenue_consistency"]["stability"] = "high" if results["revenue_consistency"]["cv"] < 15 else "medium" if results["revenue_consistency"]["cv"] < 30 else "low"
            
            # Profit consistency
            if 'net_profit' in df.columns:
                profit_std = df['net_profit'].std()
                profit_mean = df['net_profit'].mean()
                if profit_mean > 0:
                    results["profit_consistency"]["cv"] = (profit_std / profit_mean) * 100
                    results["profit_consistency"]["stability"] = "high" if results["profit_consistency"]["cv"] < 20 else "medium" if results["profit_consistency"]["cv"] < 40 else "low"
            
            # Earnings quality (operating cash flow vs. reported earnings)
            # This would require cash flow data which might not be available
            # For now, we'll look at the ratio of operating profit to net profit as a proxy
            if 'operating_profit' in df.columns and 'net_profit' in df.columns:
                df['op_to_np_ratio'] = df['operating_profit'] / df['net_profit']
                results["earnings_quality"]["op_to_np_ratio"] = df['op_to_np_ratio'].mean()
                
                # Ideal range is typically between 1.0 and 1.5
                ratio = results["earnings_quality"]["op_to_np_ratio"]
                if 1.0 <= ratio <= 1.5:
                    results["earnings_quality"]["rating"] = "high"
                elif 0.8 <= ratio < 1.0 or 1.5 < ratio <= 1.8:
                    results["earnings_quality"]["rating"] = "medium"
                else:
                    results["earnings_quality"]["rating"] = "low"
            
            # Check for sequential losses (negative profitability quarters)
            if 'net_profit' in df.columns:
                loss_quarters = (df['net_profit'] < 0).sum()
                results["profit_consistency"]["loss_quarters"] = int(loss_quarters)
                results["profit_consistency"]["profitable_ratio"] = (len(df) - loss_quarters) / len(df)
                
                # Check for sequential losses
                if len(df) >= 4:
                    sequential_losses = 0
                    max_sequential = 0
                    for i, profit in enumerate(df['net_profit']):
                        if profit < 0:
                            sequential_losses += 1
                            max_sequential = max(max_sequential, sequential_losses)
                        else:
                            sequential_losses = 0
                    
                    results["profit_consistency"]["max_sequential_losses"] = max_sequential
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing consistency metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_score(self, growth: Dict[str, Any], profitability: Dict[str, Any], 
                         consistency: Dict[str, Any]) -> float:
        """
        Calculate overall quarterly performance score.
        
        Args:
            growth: Growth analysis results
            profitability: Profitability analysis results
            consistency: Consistency analysis results
            
        Returns:
            Score from 0-100
        """
        try:
            score = 50  # Start at middle
            
            # Growth factors (up to +25 points)
            if 'revenue_growth' in growth and 'yoy_avg' in growth['revenue_growth']:
                rev_growth = growth['revenue_growth']['yoy_avg']
                # Revenue growth score (up to 10 points)
                if rev_growth > 30:
                    score += 10
                elif rev_growth > 20:
                    score += 8
                elif rev_growth > 10:
                    score += 5
                elif rev_growth > 5:
                    score += 3
                elif rev_growth > 0:
                    score += 1
                else:
                    score -= 5  # Penalty for negative growth
            
            if 'profit_growth' in growth and 'yoy_avg' in growth['profit_growth']:
                profit_growth = growth['profit_growth']['yoy_avg']
                # Profit growth score (up to 15 points)
                if profit_growth > 40:
                    score += 15
                elif profit_growth > 25:
                    score += 12
                elif profit_growth > 15:
                    score += 9
                elif profit_growth > 8:
                    score += 6
                elif profit_growth > 0:
                    score += 3
                else:
                    score -= 8  # Penalty for negative growth
            
            # Profitability factors (up to +25 points)
            if 'margins' in profitability:
                margins = profitability['margins']
                
                # Net margin (up to 15 points)
                if 'net_margin_avg' in margins:
                    net_margin = margins['net_margin_avg']
                    if net_margin > 25:
                        score += 15
                    elif net_margin > 20:
                        score += 12
                    elif net_margin > 15:
                        score += 9
                    elif net_margin > 10:
                        score += 6
                    elif net_margin > 5:
                        score += 3
                    elif net_margin <= 0:
                        score -= 10  # Penalty for negative margin
                
                # Operating margin trend (up to 10 points)
                if 'trends' in profitability and 'opm_trend_value' in profitability['trends']:
                    trend = profitability['trends']['opm_trend_value']
                    if trend > 1.5:
                        score += 10  # Strong positive trend
                    elif trend > 0.5:
                        score += 7
                    elif trend > 0:
                        score += 3
                    elif trend < -1.0:
                        score -= 7  # Strong negative trend
                    elif trend < 0:
                        score -= 3
            
            # Consistency factors (up to +25 points)
            if 'profit_consistency' in consistency:
                # Consistency of profits (up to 15 points)
                if 'profitable_ratio' in consistency['profit_consistency']:
                    prof_ratio = consistency['profit_consistency']['profitable_ratio']
                    if prof_ratio == 1.0:
                        score += 15  # All quarters profitable
                    elif prof_ratio >= 0.9:
                        score += 12
                    elif prof_ratio >= 0.8:
                        score += 8
                    elif prof_ratio >= 0.7:
                        score += 5
                    elif prof_ratio <= 0.5:
                        score -= 10  # More than half quarters unprofitable
                
                # Sequential losses (up to -15 points penalty)
                if 'max_sequential_losses' in consistency['profit_consistency']:
                    seq_losses = consistency['profit_consistency']['max_sequential_losses']
                    if seq_losses >= 3:
                        score -= 15  # 3+ quarters of sequential losses
                    elif seq_losses == 2:
                        score -= 10
                    elif seq_losses == 1:
                        score -= 5
            
            # Revenue consistency (up to 10 points)
            if 'revenue_consistency' in consistency and 'stability' in consistency['revenue_consistency']:
                stability = consistency['revenue_consistency']['stability']
                if stability == "high":
                    score += 10
                elif stability == "medium":
                    score += 5
                
            # Earnings quality (up to 10 points)
            if 'earnings_quality' in consistency and 'rating' in consistency['earnings_quality']:
                quality = consistency['earnings_quality']['rating']
                if quality == "high":
                    score += 10
                elif quality == "medium":
                    score += 5
                elif quality == "low":
                    score -= 5
            
            # Ensure score is within 0-100 range
            score = max(0, min(100, score))
            return round(score, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating quarterly score: {e}")
            return 50.0  # Default to neutral score on error
    
    def _save_analysis(self, analysis_result: Dict[str, Any]) -> bool:
        """
        Save analysis results to database.
        
        Args:
            analysis_result: Analysis results to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if we need to update an existing analysis
            existing = self.db.quarterly_analysis_collection.find_one({
                "symbol": analysis_result["symbol"],
                "exchange": analysis_result["exchange"],
                "latest_quarter": analysis_result["latest_quarter"]
            })
            
            if existing:
                # Update existing analysis
                self.db.quarterly_analysis_collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": analysis_result}
                )
            else:
                # Insert new analysis
                self.db.quarterly_analysis_collection.insert_one(analysis_result)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving quarterly analysis: {e}")
            return False
    
    def get_latest_analysis(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get the most recent quarterly analysis for a symbol.
        
        Args:
            symbol: Stock symbol/ticker
            exchange: Stock exchange
            
        Returns:
            Dictionary containing latest analysis or empty dict if not found
        """
        try:
            latest = self.db.quarterly_analysis_collection.find_one(
                {"symbol": symbol, "exchange": exchange},
                sort=[("analysis_date", -1)]
            )
            
            if latest:
                return latest
            else:
                self.logger.warning(f"No quarterly analysis found for {symbol}:{exchange}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error retrieving latest quarterly analysis: {e}")
            return {}
    
    def get_peer_comparison(self, symbol: str, exchange: str, peers: List[str]) -> Dict[str, Any]:
        """
        Compare quarterly performance with peer companies.
        
        Args:
            symbol: Stock symbol/ticker
            exchange: Stock exchange
            peers: List of peer company symbols
            
        Returns:
            Dictionary containing peer comparison metrics
        """
        try:
            # Get latest analysis for target company
            target_analysis = self.get_latest_analysis(symbol, exchange)
            
            if not target_analysis:
                return {"status": "error", "message": "No analysis found for target company"}
            
            # Get latest analysis for peer companies
            peer_analyses = []
            for peer in peers:
                peer_analysis = self.get_latest_analysis(peer, exchange)
                if peer_analysis:
                    peer_analyses.append(peer_analysis)
            
            if not peer_analyses:
                return {"status": "error", "message": "No peer analyses found"}
            
            # Calculate average peer metrics
            peer_scores = [p.get("quarterly_score", 0) for p in peer_analyses]
            peer_avg_score = sum(peer_scores) / len(peer_scores)
            
            # Get growth metrics
            peer_revenue_growth = []
            peer_profit_growth = []
            
            for p in peer_analyses:
                if "growth_analysis" in p and "revenue_growth" in p["growth_analysis"]:
                    if "yoy_avg" in p["growth_analysis"]["revenue_growth"]:
                        peer_revenue_growth.append(p["growth_analysis"]["revenue_growth"]["yoy_avg"])
                
                if "growth_analysis" in p and "profit_growth" in p["growth_analysis"]:
                    if "yoy_avg" in p["growth_analysis"]["profit_growth"]:
                        peer_profit_growth.append(p["growth_analysis"]["profit_growth"]["yoy_avg"])
            
            peer_avg_revenue_growth = sum(peer_revenue_growth) / len(peer_revenue_growth) if peer_revenue_growth else 0
            peer_avg_profit_growth = sum(peer_profit_growth) / len(peer_profit_growth) if peer_profit_growth else 0
            
            # Target company metrics
            target_revenue_growth = 0
            target_profit_growth = 0
            
            if "growth_analysis" in target_analysis and "revenue_growth" in target_analysis["growth_analysis"]:
                if "yoy_avg" in target_analysis["growth_analysis"]["revenue_growth"]:
                    target_revenue_growth = target_analysis["growth_analysis"]["revenue_growth"]["yoy_avg"]
            
            if "growth_analysis" in target_analysis and "profit_growth" in target_analysis["growth_analysis"]:
                if "yoy_avg" in target_analysis["growth_analysis"]["profit_growth"]:
                    target_profit_growth = target_analysis["growth_analysis"]["profit_growth"]["yoy_avg"]
            
            # Calculate percentile rankings
            all_scores = peer_scores + [target_analysis.get("quarterly_score", 0)]
            all_scores.sort()
            target_score = target_analysis.get("quarterly_score", 0)
            
            if all_scores:
                percentile = all_scores.index(target_score) / len(all_scores) * 100
            else:
                percentile = 50
            
            # Build comparison result
            comparison = {
                "symbol": symbol,
                "exchange": exchange,
                "peer_count": len(peer_analyses),
                "peers": [p.get("symbol") for p in peer_analyses],
                "target_score": target_analysis.get("quarterly_score", 0),
                "peer_avg_score": peer_avg_score,
                "percentile_rank": percentile,
                "growth_comparison": {
                    "target_revenue_growth": target_revenue_growth,
                    "peer_avg_revenue_growth": peer_avg_revenue_growth,
                    "target_profit_growth": target_profit_growth,
                    "peer_avg_profit_growth": peer_avg_profit_growth
                },
                "ranking": "above_average" if target_score > peer_avg_score else "below_average",
                "analysis_date": datetime.now()
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error performing peer comparison: {e}")
            return {"status": "error", "message": str(e)}


# Usage example
if __name__ == "__main__":
    # This would be used for testing only
    from pymongo import MongoClient
    
    # Example connection to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["automated_trading"]
    
    # Initialize analyzer
    analyzer = QuarterlyAnalyzer(db)
    
    # Example usage
    results = analyzer.analyze("TATASTEEL", "NSE")
    print(results)