"""
Portfolio Hedging Module

This module implements portfolio hedging strategies to:
- Reduce overall portfolio risk
- Protect against market downturns
- Maintain exposure while limiting downside
- Hedge against specific risk factors
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import math

class PortfolioHedging:
    """
    Implements portfolio hedging strategies.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the module with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Strategy parameters (configurable)
        self.params = {
            # General parameters
            "lookback_period": 90,  # Days for historical analysis
            "min_data_points": 30,  # Minimum data points needed
            
            # Hedging thresholds
            "market_hedge_threshold": 20,  # VIX level to trigger market hedge
            "sector_hedge_threshold": -5,  # Sector performance threshold (%)
            "portfolio_drawdown_threshold": -5,  # Portfolio drawdown to trigger hedge (%)
            
            # Hedge sizing
            "market_hedge_ratio": 0.5,  # Portion of portfolio to hedge
            "max_hedge_allocation": 0.2,  # Maximum allocation to hedging instruments
            "correlation_threshold": 0.7,  # Minimum correlation for effective hedge
            
            # Tactical hedging
            "vix_rsi_threshold": 30,  # VIX RSI level to trigger tactical hedge
            "put_call_ratio_threshold": 1.2,  # Put/call ratio threshold
            
            # Hedge removal
            "hedge_removal_threshold": 10,  # VIX level to remove market hedge
            "profit_take_threshold": 5,  # Take profit from hedge (%)
            "max_hedge_duration": 30,  # Maximum days to hold hedge
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
    
    def analyze_portfolio_risk(self, portfolio_id: str = None) -> Dict[str, Any]:
        """
        Analyze portfolio risk factors to determine hedging requirements.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dictionary with portfolio risk analysis
        """
        try:
            # Get portfolio data
            portfolio = self._get_portfolio_data(portfolio_id)
            
            if not portfolio or "positions" not in portfolio or not portfolio["positions"]:
                return {"status": "error", "message": "Portfolio not found or empty"}
            
            # Get current market conditions
            market_conditions = self._get_market_conditions()
            
            # Calculate portfolio beta
            portfolio_beta = self._calculate_portfolio_beta(portfolio["positions"])
            
            # Calculate sector exposure
            sector_exposure = self._calculate_sector_exposure(portfolio["positions"])
            
            # Calculate drawdown
            portfolio_drawdown = self._calculate_portfolio_drawdown(portfolio_id)
            
            # Calculate volatility
            portfolio_volatility = self._calculate_portfolio_volatility(portfolio["positions"])
            
            # Calculate correlation with market
            portfolio_correlation = self._calculate_market_correlation(portfolio["positions"])
            
            # Calculate downside risk
            downside_risk = self._calculate_downside_risk(portfolio["positions"])
            
            # Determine overall risk level
            risk_factors = []
            
            # Check market conditions
            vix_level = market_conditions.get("vix_level", 0)
            if vix_level >= self.params["market_hedge_threshold"]:
                risk_factors.append({
                    "factor": "high_market_volatility",
                    "description": f"VIX at {vix_level:.2f}, above threshold of {self.params['market_hedge_threshold']}",
                    "severity": "high"
                })
            
            # Check portfolio beta
            if portfolio_beta > 1.2:
                risk_factors.append({
                    "factor": "high_beta",
                    "description": f"Portfolio beta of {portfolio_beta:.2f} indicates high market sensitivity",
                    "severity": "medium"
                })
            
            # Check sector concentration
            for sector, exposure in sector_exposure.items():
                if exposure > 0.25:  # More than 25% in a single sector
                    risk_factors.append({
                        "factor": "sector_concentration",
                        "description": f"{sector} sector concentration of {exposure:.1%}",
                        "severity": "medium"
                    })
            
            # Check drawdown
            if portfolio_drawdown < self.params["portfolio_drawdown_threshold"]:
                risk_factors.append({
                    "factor": "significant_drawdown",
                    "description": f"Portfolio drawdown of {portfolio_drawdown:.1%}",
                    "severity": "high"
                })
            
            # Check volatility
            if portfolio_volatility > 20:  # Annualized volatility > 20%
                risk_factors.append({
                    "factor": "high_volatility",
                    "description": f"Portfolio volatility of {portfolio_volatility:.1f}%",
                    "severity": "medium"
                })
            
            # Determine if hedging is recommended
            hedge_recommended = False
            if any(factor["severity"] == "high" for factor in risk_factors):
                hedge_recommended = True
            elif len(risk_factors) >= 2:
                hedge_recommended = True
            
            # Calculate portfolio value
            portfolio_value = sum(pos.get("current_value", 0) for pos in portfolio["positions"])
            
            return {
                "portfolio_id": portfolio_id,
                "portfolio_value": portfolio_value,
                "portfolio_beta": portfolio_beta,
                "sector_exposure": sector_exposure,
                "portfolio_drawdown": portfolio_drawdown,
                "portfolio_volatility": portfolio_volatility,
                "portfolio_correlation": portfolio_correlation,
                "downside_risk": downside_risk,
                "market_conditions": market_conditions,
                "risk_factors": risk_factors,
                "hedge_recommended": hedge_recommended,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio risk: {e}")
            return {"status": "error", "error": str(e)}
    
    def recommend_hedging_strategy(self, portfolio_id: str = None) -> Dict[str, Any]:
        """
        Recommend portfolio hedging strategy.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dictionary with hedging strategy recommendation
        """
        try:
            # Analyze portfolio risk
            risk_analysis = self.analyze_portfolio_risk(portfolio_id)
            
            if "status" in risk_analysis and risk_analysis["status"] == "error":
                return risk_analysis
            
            # Check if hedging is recommended
            if not risk_analysis.get("hedge_recommended", False):
                return {
                    "status": "no_hedge_needed",
                    "message": "Portfolio hedging is not currently recommended",
                    "risk_analysis": risk_analysis
                }
            
            # Get portfolio data
            portfolio = self._get_portfolio_data(portfolio_id)
            
            # Calculate portfolio value
            portfolio_value = risk_analysis.get("portfolio_value", 0)
            
            # Determine hedging type needed
            hedge_strategies = []
            hedge_instruments = []
            
            # Identify risk factors
            risk_factors = risk_analysis.get("risk_factors", [])
            risk_factor_types = [factor["factor"] for factor in risk_factors]
            
            # 1. Market hedge strategies
            if "high_market_volatility" in risk_factor_types or "high_beta" in risk_factor_types:
                # Index hedge recommendation
                index_hedge = self._recommend_index_hedge(risk_analysis)
                if index_hedge:
                    hedge_strategies.append(index_hedge)
                    hedge_instruments.append(index_hedge.get("instrument"))
                
                # VIX-based hedge recommendation
                vix_hedge = self._recommend_vix_hedge(risk_analysis)
                if vix_hedge:
                    hedge_strategies.append(vix_hedge)
                    hedge_instruments.append(vix_hedge.get("instrument"))
            
            # 2. Sector hedge strategies
            if "sector_concentration" in risk_factor_types:
                # Identify concentrated sectors
                concentrated_sectors = []
                for sector, exposure in risk_analysis.get("sector_exposure", {}).items():
                    if exposure > 0.25:  # More than 25% in a single sector
                        concentrated_sectors.append(sector)
                
                # Get sector hedge recommendations
                for sector in concentrated_sectors:
                    sector_hedge = self._recommend_sector_hedge(risk_analysis, sector)
                    if sector_hedge:
                        hedge_strategies.append(sector_hedge)
                        hedge_instruments.append(sector_hedge.get("instrument"))
            
            # 3. Volatility hedge strategies
            if "high_volatility" in risk_factor_types:
                # Long volatility recommendation
                vol_hedge = self._recommend_volatility_hedge(risk_analysis)
                if vol_hedge:
                    hedge_strategies.append(vol_hedge)
                    hedge_instruments.append(vol_hedge.get("instrument"))
            
            # 4. Drawdown protection strategies
            if "significant_drawdown" in risk_factor_types:
                # Stop-loss recommendation
                stop_hedge = self._recommend_stop_loss_protection(risk_analysis)
                if stop_hedge:
                    hedge_strategies.append(stop_hedge)
            
            # Calculate total hedge allocation
            total_hedge_value = sum(strategy.get("allocation_value", 0) for strategy in hedge_strategies)
            hedge_ratio = total_hedge_value / portfolio_value if portfolio_value > 0 else 0
            
            # Check if hedge ratio is within limits
            if hedge_ratio > self.params["max_hedge_allocation"]:
                # Scale down allocations to meet maximum
                scale_factor = self.params["max_hedge_allocation"] / hedge_ratio
                for strategy in hedge_strategies:
                    strategy["allocation_value"] = strategy["allocation_value"] * scale_factor
                    strategy["allocation_percent"] = strategy["allocation_percent"] * scale_factor
                
                total_hedge_value = sum(strategy.get("allocation_value", 0) for strategy in hedge_strategies)
                hedge_ratio = self.params["max_hedge_allocation"]
            
            # Calculate hedge impact
            current_beta = risk_analysis.get("portfolio_beta", 1.0)
            estimated_new_beta = self._estimate_hedged_beta(current_beta, hedge_strategies)
            
            current_volatility = risk_analysis.get("portfolio_volatility", 0)
            estimated_new_volatility = self._estimate_hedged_volatility(current_volatility, hedge_strategies)
            
            # Set hedge exit conditions
            exit_conditions = self._generate_hedge_exit_conditions(risk_analysis, hedge_strategies)
            
            return {
                "portfolio_id": portfolio_id,
                "hedge_recommended": True,
                "hedge_strategies": hedge_strategies,
                "hedge_instruments": hedge_instruments,
                "total_hedge_value": total_hedge_value,
                "hedge_ratio": hedge_ratio,
                "current_beta": current_beta,
                "estimated_new_beta": estimated_new_beta,
                "current_volatility": current_volatility,
                "estimated_new_volatility": estimated_new_volatility,
                "exit_conditions": exit_conditions,
                "risk_analysis": risk_analysis,
                "recommendation_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error recommending hedging strategy: {e}")
            return {"status": "error", "error": str(e)}
    
    def optimize_hedge_allocation(self, portfolio_id: str = None) -> Dict[str, Any]:
        """
        Optimize the allocation to different hedging instruments.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dictionary with optimized hedge allocation
        """
        try:
            # Get hedging recommendation
            recommendation = self.recommend_hedging_strategy(portfolio_id)
            
            if "status" in recommendation and recommendation["status"] != "hedge_recommended":
                return recommendation
            
            # Get portfolio data
            portfolio = self._get_portfolio_data(portfolio_id)
            
            # Calculate portfolio value
            portfolio_value = recommendation.get("risk_analysis", {}).get("portfolio_value", 0)
            
            # Get risk factors
            risk_factors = recommendation.get("risk_analysis", {}).get("risk_factors", [])
            
            # Get hedge strategies
            hedge_strategies = recommendation.get("hedge_strategies", [])
            
            # Define optimization weights based on risk factors
            weights = {
                "high_market_volatility": 0.4,
                "high_beta": 0.3,
                "sector_concentration": 0.2,
                "high_volatility": 0.2,
                "significant_drawdown": 0.3
            }
            
            # Calculate total weight for present risk factors
            risk_factor_types = [factor["factor"] for factor in risk_factors]
            total_weight = sum(weights[factor] for factor in risk_factor_types if factor in weights)
            
            # Normalize weights
            if total_weight > 0:
                normalized_weights = {factor: weights[factor] / total_weight for factor in risk_factor_types if factor in weights}
            else:
                normalized_weights = {factor: 1.0 / len(risk_factor_types) for factor in risk_factor_types}
            
            # Assign allocation based on risk factor weights
            optimized_strategies = []
            max_allocation = self.params["max_hedge_allocation"] * portfolio_value
            allocated_value = 0
            
            for strategy in hedge_strategies:
                strategy_type = strategy.get("strategy_type")
                
                # Map strategy types to risk factors
                if strategy_type == "index_hedge":
                    relevant_factors = ["high_market_volatility", "high_beta"]
                elif strategy_type == "vix_hedge":
                    relevant_factors = ["high_market_volatility", "high_volatility"]
                elif strategy_type == "sector_hedge":
                    relevant_factors = ["sector_concentration"]
                elif strategy_type == "volatility_hedge":
                    relevant_factors = ["high_volatility"]
                elif strategy_type == "stop_loss":
                    relevant_factors = ["significant_drawdown"]
                else:
                    relevant_factors = []
                
                # Calculate strategy weight based on relevant risk factors
                strategy_weight = sum(normalized_weights.get(factor, 0) for factor in relevant_factors)
                
                # Assign allocation
                allocation_value = min(strategy_weight * max_allocation, max_allocation - allocated_value)
                allocation_percent = allocation_value / portfolio_value if portfolio_value > 0 else 0
                
                # Update strategy allocation
                optimized_strategy = strategy.copy()
                optimized_strategy["allocation_value"] = allocation_value
                optimized_strategy["allocation_percent"] = allocation_percent
                optimized_strategy["optimization_weight"] = strategy_weight
                
                optimized_strategies.append(optimized_strategy)
                allocated_value += allocation_value
            
            # Calculate total allocated hedge
            total_hedge_value = sum(strategy.get("allocation_value", 0) for strategy in optimized_strategies)
            hedge_ratio = total_hedge_value / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate optimized impact
            current_beta = recommendation.get("current_beta", 1.0)
            optimized_beta = self._estimate_hedged_beta(current_beta, optimized_strategies)
            
            current_volatility = recommendation.get("current_volatility", 0)
            optimized_volatility = self._estimate_hedged_volatility(current_volatility, optimized_strategies)
            
            # Set hedge exit conditions
            exit_conditions = self._generate_hedge_exit_conditions(recommendation.get("risk_analysis", {}), optimized_strategies)
            
            return {
                "portfolio_id": portfolio_id,
                "optimized_strategies": optimized_strategies,
                "total_hedge_value": total_hedge_value,
                "hedge_ratio": hedge_ratio,
                "current_beta": current_beta,
                "optimized_beta": optimized_beta,
                "current_volatility": current_volatility,
                "optimized_volatility": optimized_volatility,
                "exit_conditions": exit_conditions,
                "risk_analysis": recommendation.get("risk_analysis", {}),
                "optimization_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing hedge allocation: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_hedging_orders(self, portfolio_id: str = None) -> Dict[str, Any]:
        """
        Generate orders for implementing the hedging strategy.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dictionary with hedging orders
        """
        try:
            # Get optimized hedging allocation
            optimization = self.optimize_hedge_allocation(portfolio_id)
            
            if "status" in optimization and optimization["status"] != "hedge_recommended":
                return optimization
            
            # Get portfolio value
            portfolio_value = optimization.get("risk_analysis", {}).get("portfolio_value", 0)
            
            # Get optimized strategies
            strategies = optimization.get("optimized_strategies", [])
            
            # Generate orders
            orders = []
            
            for strategy in strategies:
                strategy_type = strategy.get("strategy_type")
                instrument = strategy.get("instrument", {})
                allocation_value = strategy.get("allocation_value", 0)
                
                if strategy_type == "stop_loss":
                    # Stop-loss isn't an order to place, it's a recommendation to set stops
                    continue
                
                if not instrument or allocation_value <= 0:
                    continue
                
                # Calculate quantity
                symbol = instrument.get("symbol")
                current_price = instrument.get("price", 0)
                
                if current_price <= 0:
                    continue
                
                quantity = self._calculate_hedge_quantity(allocation_value, current_price, symbol, strategy_type)
                
                if quantity <= 0:
                    continue
                
                # Create order
                order = {
                    "portfolio_id": portfolio_id,
                    "strategy_type": strategy_type,
                    "order_type": "market",
                    "direction": "buy",  # Hedges are typically long positions
                    "symbol": symbol,
                    "quantity": quantity,
                    "estimated_price": current_price,
                    "estimated_value": quantity * current_price,
                    "allocation_target": allocation_value,
                    "hedge_reason": strategy.get("reason"),
                    "exit_conditions": [
                        condition for condition in optimization.get("exit_conditions", [])
                        if condition.get("strategy_type") == strategy_type
                    ],
                    "order_status": "pending",
                    "created_at": datetime.now()
                }
                
                orders.append(order)
            
            # Calculate total order value
            total_order_value = sum(order.get("estimated_value", 0) for order in orders)
            hedge_ratio = total_order_value / portfolio_value if portfolio_value > 0 else 0
            
            return {
                "portfolio_id": portfolio_id,
                "orders": orders,
                "total_order_value": total_order_value,
                "hedge_ratio": hedge_ratio,
                "optimization": optimization,
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating hedging orders: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_hedge_exit_conditions(self, portfolio_id: str = None) -> Dict[str, Any]:
        """
        Check if exit conditions for any hedges have been met.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dictionary with hedge exit recommendations
        """
        try:
            # Get portfolio data
            portfolio = self._get_portfolio_data(portfolio_id)
            
            if not portfolio or "positions" not in portfolio or not portfolio["positions"]:
                return {"status": "error", "message": "Portfolio not found or empty"}
            
            # Get current market conditions
            market_conditions = self._get_market_conditions()
            
            # Identify hedge positions
            hedge_positions = [pos for pos in portfolio["positions"] if pos.get("is_hedge", False)]
            
            if not hedge_positions:
                return {
                    "status": "no_hedges",
                    "message": "No active hedge positions found"
                }
            
            # Check exit conditions for each hedge
            exit_recommendations = []
            
            for position in hedge_positions:
                symbol = position.get("symbol")
                strategy_type = position.get("strategy_type")
                entry_price = position.get("entry_price", 0)
                current_price = position.get("current_price", 0)
                entry_date = position.get("entry_date", datetime.now() - timedelta(days=30))
                
                # Calculate profit/loss
                profit_loss_percent = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
                
                # Check exit conditions
                exit_reasons = []
                
                # 1. Take profit condition
                if profit_loss_percent >= self.params["profit_take_threshold"]:
                    exit_reasons.append({
                        "reason": "profit_target_reached",
                        "description": f"Hedge profit of {profit_loss_percent:.1f}% exceeds threshold of {self.params['profit_take_threshold']}%"
                    })
                
                # 2. Time-based condition
                days_held = (datetime.now() - entry_date).days
                if days_held >= self.params["max_hedge_duration"]:
                    exit_reasons.append({
                        "reason": "max_duration_reached",
                        "description": f"Hedge held for {days_held} days, exceeding maximum of {self.params['max_hedge_duration']} days"
                    })
                
                # 3. Market condition change
                if strategy_type == "index_hedge" or strategy_type == "vix_hedge":
                    vix_level = market_conditions.get("vix_level", 0)
                    if vix_level <= self.params["hedge_removal_threshold"]:
                        exit_reasons.append({
                            "reason": "market_conditions_improved",
                            "description": f"VIX level {vix_level:.1f} below threshold of {self.params['hedge_removal_threshold']}"
                        })
                
                # 4. Specific hedge exit conditions
                if strategy_type == "sector_hedge":
                    sector = position.get("sector")
                    if sector:
                        sector_performance = self._get_sector_performance(sector, exchange=position.get("exchange", "NSE"))
                        if sector_performance > 0:
                            exit_reasons.append({
                                "reason": "sector_performance_improved",
                                "description": f"{sector} sector performance improved to {sector_performance:.1f}%"
                            })
                
                # If any exit conditions are met, recommend exit
                if exit_reasons:
                    exit_recommendations.append({
                        "symbol": symbol,
                        "strategy_type": strategy_type,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "profit_loss_percent": profit_loss_percent,
                        "days_held": days_held,
                        "exit_reasons": exit_reasons,
                        "recommendation": "exit"
                    })
            
            return {
                "portfolio_id": portfolio_id,
                "exit_recommendations": exit_recommendations,
                "market_conditions": market_conditions,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error checking hedge exit conditions: {e}")
            return {"status": "error", "error": str(e)}
    
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
            
            return portfolio or {}
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio data: {e}")
            return {}
    
    def _get_market_conditions(self) -> Dict[str, Any]:
        """
        Get current market conditions.
        
        Returns:
            Dictionary with market conditions
        """
        try:
            # Get VIX data
            vix_data = self._get_market_data("VIX", "NSE", days=30) or self._get_market_data("INDIA VIX", "NSE", days=30)
            
            vix_level = None
            vix_percentile = None
            vix_trend = None
            
            if vix_data:
                df = pd.DataFrame(vix_data).sort_values("timestamp")
                
                if not df.empty:
                    vix_level = df['close'].iloc[-1]
                    
                    # Calculate VIX percentile
                    vix_percentile = (df['close'] < vix_level).mean() * 100
                    
                    # Determine VIX trend
                    vix_trend = "rising" if df['close'].iloc[-1] > df['close'].iloc[-5] else "falling"
                    
                    # Calculate VIX RSI
                    if len(df) >= 14:
                        delta = df['close'].diff()
                        up = delta.clip(lower=0)
                        down = -1 * delta.clip(upper=0)
                        
                        avg_up = up.rolling(window=14).mean()
                        avg_down = down.rolling(window=14).mean()
                        
                        if avg_down.iloc[-1] != 0:
                            rs = avg_up.iloc[-1] / avg_down.iloc[-1]
                            vix_rsi = 100 - (100 / (1 + rs))
                        else:
                            vix_rsi = 100
                    else:
                        vix_rsi = None
            
            # Get market index data
            index_data = self._get_market_data("NIFTY", "NSE", days=30)
            
            market_trend = None
            market_performance_1d = None
            market_performance_5d = None
            
            if index_data:
                df = pd.DataFrame(index_data).sort_values("timestamp")
                
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    prev_day_price = df['close'].iloc[-2] if len(df) > 1 else None
                    five_day_price = df['close'].iloc[-6] if len(df) > 5 else None
                    
                    if prev_day_price:
                        market_performance_1d = (current_price / prev_day_price - 1) * 100
                    
                    if five_day_price:
                        market_performance_5d = (current_price / five_day_price - 1) * 100
                    
                    # Determine market trend
                    ma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
                    ma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
                    
                    if ma_20 and ma_50:
                        if current_price > ma_20 and current_price > ma_50:
                            market_trend = "bullish"
                        elif current_price < ma_20 and current_price < ma_50:
                            market_trend = "bearish"
                        else:
                            market_trend = "neutral"
            
            # Get put-call ratio data
            pcr_data = self._get_put_call_ratio()
            
            return {
                "vix_level": vix_level,
                "vix_percentile": vix_percentile,
                "vix_trend": vix_trend,
                "vix_rsi": vix_rsi,
                "market_trend": market_trend,
                "market_performance_1d": market_performance_1d,
                "market_performance_5d": market_performance_5d,
                "put_call_ratio": pcr_data.get("put_call_ratio"),
                "pcr_percentile": pcr_data.get("pcr_percentile"),
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market conditions: {e}")
            return {}
    
    def _calculate_portfolio_beta(self, positions: List[Dict[str, Any]]) -> float:
        """
        Calculate portfolio beta.
        
        Args:
            positions: List of portfolio positions
            
        Returns:
            Portfolio beta
        """
        try:
            # Default if no data
            if not positions:
                return 1.0
            
            # Calculate weighted beta
            total_value = sum(pos.get("current_value", 0) for pos in positions)
            weighted_beta = 0
            
            for position in positions:
                beta = position.get("beta", 1.0)  # Default to 1.0 if not available
                position_value = position.get("current_value", 0)
                
                # Apply weight
                weighted_beta += beta * (position_value / total_value) if total_value > 0 else 0
            
            return weighted_beta
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0
    
    def _calculate_sector_exposure(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate sector exposure.
        
        Args:
            positions: List of portfolio positions
            
        Returns:
            Dictionary with sector exposure ratios
        """
        try:
            sector_values = {}
            total_value = sum(pos.get("current_value", 0) for pos in positions)
            
            if total_value <= 0:
                return {}
            
            # Calculate value by sector
            # Calculate value by sector
            for position in positions:
                sector = position.get("sector", "Unknown")
                position_value = position.get("current_value", 0)
                
                if sector not in sector_values:
                    sector_values[sector] = 0
                
                sector_values[sector] += position_value
            
            # Calculate exposure ratios
            sector_exposure = {sector: value / total_value for sector, value in sector_values.items()}
            
            return sector_exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating sector exposure: {e}")
            return {}
    
    def _calculate_portfolio_drawdown(self, portfolio_id: str = None) -> float:
        """
        Calculate portfolio drawdown.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Portfolio drawdown percentage
        """
        try:
            # Get portfolio history
            history = self._get_portfolio_history(portfolio_id)
            
            if not history or len(history) < 2:
                return 0.0
            
            # Convert to dataframe
            df = pd.DataFrame(history)
            
            # Get current value
            current_value = df['value'].iloc[-1]
            
            # Calculate peak value
            peak_value = df['value'].max()
            
            # Calculate drawdown
            drawdown = (current_value / peak_value - 1) * 100
            
            return drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio drawdown: {e}")
            return 0.0
    
    def _calculate_portfolio_volatility(self, positions: List[Dict[str, Any]]) -> float:
        """
        Calculate portfolio volatility.
        
        Args:
            positions: List of portfolio positions
            
        Returns:
            Portfolio volatility (annualized %)
        """
        try:
            # Get symbols from positions
            symbols = [pos.get("symbol") for pos in positions if pos.get("symbol")]
            exchanges = [pos.get("exchange", "NSE") for pos in positions if pos.get("symbol")]
            
            if not symbols:
                return 0.0
            
            # Get historical data for each symbol
            price_data = {}
            
            for i, symbol in enumerate(symbols):
                exchange = exchanges[i] if i < len(exchanges) else "NSE"
                data = self._get_market_data(symbol, exchange, days=60)
                
                if data and len(data) > 30:
                    df = pd.DataFrame(data).sort_values("timestamp")
                    price_data[symbol] = df['close'].values
            
            if not price_data:
                return 0.0
            
            # Calculate returns
            returns_data = {}
            for symbol, prices in price_data.items():
                returns = np.diff(prices) / prices[:-1]
                returns_data[symbol] = returns
            
            # Convert to dataframe for easier handling
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate portfolio weights
            total_value = sum(pos.get("current_value", 0) for pos in positions if pos.get("symbol") in returns_df.columns)
            weights = {}
            
            for position in positions:
                symbol = position.get("symbol")
                if symbol in returns_df.columns:
                    position_value = position.get("current_value", 0)
                    weights[symbol] = position_value / total_value if total_value > 0 else 0
            
            # Calculate weighted returns
            portfolio_returns = np.zeros(len(returns_df))
            
            for symbol, weight in weights.items():
                if symbol in returns_df.columns:
                    portfolio_returns += returns_df[symbol].values * weight
            
            # Calculate volatility (annualized)
            volatility = np.std(portfolio_returns) * np.sqrt(252) * 100
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0
    
    def _calculate_market_correlation(self, positions: List[Dict[str, Any]]) -> float:
        """
        Calculate portfolio correlation with market.
        
        Args:
            positions: List of portfolio positions
            
        Returns:
            Correlation coefficient
        """
        try:
            # Get symbols from positions
            symbols = [pos.get("symbol") for pos in positions if pos.get("symbol")]
            exchanges = [pos.get("exchange", "NSE") for pos in positions if pos.get("symbol")]
            
            if not symbols:
                return 1.0
            
            # Get index data
            index_data = self._get_market_data("NIFTY", "NSE", days=60)
            
            if not index_data or len(index_data) < 30:
                return 1.0
            
            index_df = pd.DataFrame(index_data).sort_values("timestamp")
            index_returns = np.diff(index_df['close'].values) / index_df['close'].values[:-1]
            
            # Get historical data for each symbol
            returns_data = {}
            
            for i, symbol in enumerate(symbols):
                exchange = exchanges[i] if i < len(exchanges) else "NSE"
                data = self._get_market_data(symbol, exchange, days=60)
                
                if data and len(data) >= len(index_data):
                    df = pd.DataFrame(data).sort_values("timestamp")
                    returns = np.diff(df['close'].values) / df['close'].values[:-1]
                    returns_data[symbol] = returns[:len(index_returns)]  # Ensure same length
            
            if not returns_data:
                return 1.0
            
            # Convert to dataframe for easier handling
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate portfolio weights
            total_value = sum(pos.get("current_value", 0) for pos in positions if pos.get("symbol") in returns_df.columns)
            weights = {}
            
            for position in positions:
                symbol = position.get("symbol")
                if symbol in returns_df.columns:
                    position_value = position.get("current_value", 0)
                    weights[symbol] = position_value / total_value if total_value > 0 else 0
            
            # Calculate weighted returns
            portfolio_returns = np.zeros(len(returns_df))
            
            for symbol, weight in weights.items():
                if symbol in returns_df.columns:
                    portfolio_returns += returns_df[symbol].values * weight
            
            # Calculate correlation
            correlation = np.corrcoef(portfolio_returns, index_returns[:len(portfolio_returns)])[0, 1]
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"Error calculating market correlation: {e}")
            return 1.0
    
    def _calculate_downside_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate portfolio downside risk metrics.
        
        Args:
            positions: List of portfolio positions
            
        Returns:
            Dictionary with downside risk metrics
        """
        try:
            # Get symbols from positions
            symbols = [pos.get("symbol") for pos in positions if pos.get("symbol")]
            exchanges = [pos.get("exchange", "NSE") for pos in positions if pos.get("symbol")]
            
            if not symbols:
                return {"downside_deviation": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0}
            
            # Get historical data for each symbol
            price_data = {}
            
            for i, symbol in enumerate(symbols):
                exchange = exchanges[i] if i < len(exchanges) else "NSE"
                data = self._get_market_data(symbol, exchange, days=252)  # Full year for downside metrics
                
                if data and len(data) > 60:
                    df = pd.DataFrame(data).sort_values("timestamp")
                    price_data[symbol] = df['close'].values
            
            if not price_data:
                return {"downside_deviation": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0}
            
            # Calculate returns
            returns_data = {}
            for symbol, prices in price_data.items():
                returns = np.diff(prices) / prices[:-1]
                returns_data[symbol] = returns
            
            # Convert to dataframe for easier handling
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate portfolio weights
            total_value = sum(pos.get("current_value", 0) for pos in positions if pos.get("symbol") in returns_df.columns)
            weights = {}
            
            for position in positions:
                symbol = position.get("symbol")
                if symbol in returns_df.columns:
                    position_value = position.get("current_value", 0)
                    weights[symbol] = position_value / total_value if total_value > 0 else 0
            
            # Calculate weighted returns
            portfolio_returns = np.zeros(len(returns_df))
            
            for symbol, weight in weights.items():
                if symbol in returns_df.columns:
                    portfolio_returns += returns_df[symbol].values * weight
            
            # Calculate downside deviation (considering returns below 0 as negative)
            negative_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = np.sqrt(np.mean(negative_returns**2)) * np.sqrt(252) * 100 if len(negative_returns) > 0 else 0
            
            # Calculate Sortino ratio (assuming risk-free rate of 5%)
            risk_free_rate = 0.05 / 252  # Daily risk-free rate
            excess_return = np.mean(portfolio_returns) - risk_free_rate
            sortino_ratio = excess_return / (downside_deviation / 100) if downside_deviation > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            peak_values = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns / peak_values - 1) * 100
            max_drawdown = abs(np.min(drawdowns))
            
            return {
                "downside_deviation": downside_deviation,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating downside risk: {e}")
            return {"downside_deviation": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0}
    
    def _recommend_index_hedge(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend index-based hedge.
        
        Args:
            risk_analysis: Portfolio risk analysis
            
        Returns:
            Dictionary with index hedge recommendation
        """
        try:
            # Get portfolio value
            portfolio_value = risk_analysis.get("portfolio_value", 0)
            portfolio_beta = risk_analysis.get("portfolio_beta", 1.0)
            
            # Get appropriate index ETF/futures for hedging
            index_instrument = self._get_index_hedging_instrument()
            
            if not index_instrument:
                return {}
            
            # Calculate allocation
            hedge_ratio = self.params["market_hedge_ratio"]
            
            # Adjust for beta (higher beta needs more hedging)
            if portfolio_beta > 1.0:
                hedge_ratio = min(hedge_ratio * portfolio_beta, 0.8)  # Cap at 80%
            
            allocation_value = portfolio_value * hedge_ratio
            allocation_percent = hedge_ratio
            
            # Get index name based on exchange
            index_name = "NIFTY 50" if index_instrument.get("exchange") == "NSE" else "S&P 500"
            
            # Build recommendation
            return {
                "strategy_type": "index_hedge",
                "instrument": index_instrument,
                "allocation_value": allocation_value,
                "allocation_percent": allocation_percent,
                "expected_impact": {
                    "beta_reduction": portfolio_beta * (1 - hedge_ratio),
                    "volatility_reduction": risk_analysis.get("portfolio_volatility", 0) * 0.5  # Estimated 50% reduction
                },
                "reason": f"Market volatility hedge using {index_name} {index_instrument.get('instrument_type', 'ETF')}"
            }
            
        except Exception as e:
            self.logger.error(f"Error recommending index hedge: {e}")
            return {}
    
    def _recommend_vix_hedge(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend VIX-based hedge.
        
        Args:
            risk_analysis: Portfolio risk analysis
            
        Returns:
            Dictionary with VIX hedge recommendation
        """
        try:
            # Get portfolio value
            portfolio_value = risk_analysis.get("portfolio_value", 0)
            
            # Get appropriate VIX ETF/futures for hedging
            vix_instrument = self._get_vix_hedging_instrument()
            
            if not vix_instrument:
                return {}
            
            # Calculate allocation (typically lower than index hedge)
            hedge_ratio = self.params["market_hedge_ratio"] * 0.3  # 30% of index hedge allocation
            allocation_value = portfolio_value * hedge_ratio
            allocation_percent = hedge_ratio
            
            # Build recommendation
            return {
                "strategy_type": "vix_hedge",
                "instrument": vix_instrument,
                "allocation_value": allocation_value,
                "allocation_percent": allocation_percent,
                "expected_impact": {
                    "beta_reduction": 0.1,  # Small beta impact
                    "volatility_reduction": risk_analysis.get("portfolio_volatility", 0) * 0.2  # Estimated 20% reduction
                },
                "reason": f"Volatility spike protection using {vix_instrument.get('name', 'VIX instrument')}"
            }
            
        except Exception as e:
            self.logger.error(f"Error recommending VIX hedge: {e}")
            return {}
    
    def _recommend_sector_hedge(self, risk_analysis: Dict[str, Any], sector: str) -> Dict[str, Any]:
        """
        Recommend sector-based hedge.
        
        Args:
            risk_analysis: Portfolio risk analysis
            sector: Sector to hedge
            
        Returns:
            Dictionary with sector hedge recommendation
        """
        try:
            # Get portfolio value
            portfolio_value = risk_analysis.get("portfolio_value", 0)
            
            # Get sector exposure
            sector_exposure = risk_analysis.get("sector_exposure", {}).get(sector, 0)
            
            if sector_exposure <= 0:
                return {}
            
            # Get appropriate sector ETF for hedging
            sector_instrument = self._get_sector_hedging_instrument(sector)
            
            if not sector_instrument:
                return {}
            
            # Calculate allocation
            sector_value = portfolio_value * sector_exposure
            hedge_ratio = 0.5  # Hedge half of sector exposure
            
            allocation_value = sector_value * hedge_ratio
            allocation_percent = sector_exposure * hedge_ratio
            
            # Build recommendation
            return {
                "strategy_type": "sector_hedge",
                "sector": sector,
                "instrument": sector_instrument,
                "allocation_value": allocation_value,
                "allocation_percent": allocation_percent,
                "expected_impact": {
                    "sector_exposure_reduction": sector_exposure * (1 - hedge_ratio)
                },
                "reason": f"Sector concentration hedge for {sector} using {sector_instrument.get('name', 'sector ETF')}"
            }
            
        except Exception as e:
            self.logger.error(f"Error recommending sector hedge: {e}")
            return {}
    
    def _recommend_volatility_hedge(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend long volatility hedge.
        
        Args:
            risk_analysis: Portfolio risk analysis
            
        Returns:
            Dictionary with volatility hedge recommendation
        """
        try:
            # Similar to VIX hedge but focused specifically on volatility
            # Get portfolio value
            portfolio_value = risk_analysis.get("portfolio_value", 0)
            
            # Get volatility ETF/option for hedging
            volatility_instrument = self._get_volatility_hedging_instrument()
            
            if not volatility_instrument:
                return {}
            
            # Calculate allocation (small allocation for volatility hedge)
            hedge_ratio = 0.05  # 5% allocation
            allocation_value = portfolio_value * hedge_ratio
            allocation_percent = hedge_ratio
            
            # Build recommendation
            return {
                "strategy_type": "volatility_hedge",
                "instrument": volatility_instrument,
                "allocation_value": allocation_value,
                "allocation_percent": allocation_percent,
                "expected_impact": {
                    "tail_risk_reduction": "high",
                    "volatility_reduction": risk_analysis.get("portfolio_volatility", 0) * 0.15  # Estimated 15% reduction
                },
                "reason": f"Tail risk protection using {volatility_instrument.get('name', 'volatility instrument')}"
            }
            
        except Exception as e:
            self.logger.error(f"Error recommending volatility hedge: {e}")
            return {}
    
    def _recommend_stop_loss_protection(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend stop-loss protection.
        
        Args:
            risk_analysis: Portfolio risk analysis
            
        Returns:
            Dictionary with stop-loss recommendation
        """
        try:
            # Get positions from risk analysis
            portfolio = self._get_portfolio_data(risk_analysis.get("portfolio_id"))
            positions = portfolio.get("positions", [])
            
            # Filter for non-hedge positions
            regular_positions = [pos for pos in positions if not pos.get("is_hedge", False)]
            
            # Calculate stop levels
            stop_recommendations = []
            
            for position in regular_positions:
                symbol = position.get("symbol")
                entry_price = position.get("entry_price", 0)
                current_price = position.get("current_price", 0)
                
                if current_price <= 0:
                    continue
                
                # Get volatility for this symbol
                volatility = self._get_symbol_volatility(symbol, position.get("exchange", "NSE"))
                
                # Calculate stop level based on volatility
                if volatility:
                    # Higher volatility needs wider stops
                    stop_percent = min(volatility * 2, 10)  # Cap at 10%
                else:
                    # Default stop percentage
                    stop_percent = 7  # 7% below current price
                
                stop_price = current_price * (1 - stop_percent / 100)
                
                stop_recommendations.append({
                    "symbol": symbol,
                    "current_price": current_price,
                    "stop_price": stop_price,
                    "stop_percent": stop_percent,
                    "volatility": volatility
                })
            
            # Build recommendation
            return {
                "strategy_type": "stop_loss",
                "stop_recommendations": stop_recommendations,
                "reason": "Portfolio drawdown protection using strategic stop-loss levels"
            }
            
        except Exception as e:
            self.logger.error(f"Error recommending stop-loss protection: {e}")
            return {}
    
    def _estimate_hedged_beta(self, current_beta: float, hedge_strategies: List[Dict[str, Any]]) -> float:
        """
        Estimate new portfolio beta after implementing hedges.
        
        Args:
            current_beta: Current portfolio beta
            hedge_strategies: List of hedge strategies
            
        Returns:
            Estimated new beta
        """
        try:
            # Start with current beta
            new_beta = current_beta
            
            # Apply impact from each hedge
            for strategy in hedge_strategies:
                strategy_type = strategy.get("strategy_type")
                
                # Different hedge types affect beta differently
                if strategy_type == "index_hedge":
                    # Index hedge directly reduces beta
                    beta_reduction = strategy.get("expected_impact", {}).get("beta_reduction", 0)
                    if beta_reduction:
                        new_beta = beta_reduction
                    else:
                        # Apply allocation-based reduction
                        allocation = strategy.get("allocation_percent", 0)
                        new_beta = current_beta * (1 - allocation)
                
                elif strategy_type == "vix_hedge" or strategy_type == "volatility_hedge":
                    # Volatility hedges have smaller beta impact
                    allocation = strategy.get("allocation_percent", 0)
                    # Estimated 20% beta reduction based on allocation
                    new_beta = new_beta * (1 - allocation * 0.2)
                
                elif strategy_type == "sector_hedge":
                    # Sector hedges reduce beta proportionally to their sector allocation
                    allocation = strategy.get("allocation_percent", 0)
                    # Estimated 50% beta reduction for the hedged portion
                    new_beta = new_beta * (1 - allocation * 0.5)
            
            return max(new_beta, 0.1)  # Ensure beta doesn't go below 0.1
            
        except Exception as e:
            self.logger.error(f"Error estimating hedged beta: {e}")
            return current_beta
    
    def _estimate_hedged_volatility(self, current_volatility: float, hedge_strategies: List[Dict[str, Any]]) -> float:
        """
        Estimate new portfolio volatility after implementing hedges.
        
        Args:
            current_volatility: Current portfolio volatility
            hedge_strategies: List of hedge strategies
            
        Returns:
            Estimated new volatility
        """
        try:
            # Start with current volatility
            new_volatility = current_volatility
            
            # Track total volatility reduction (capped)
            total_reduction = 0
            max_reduction = 0.6  # Maximum 60% volatility reduction
            
            # Apply impact from each hedge
            for strategy in hedge_strategies:
                strategy_type = strategy.get("strategy_type")
                
                # Different hedge types affect volatility differently
                if strategy_type == "index_hedge":
                    # Index hedge significantly reduces volatility
                    vol_reduction = strategy.get("expected_impact", {}).get("volatility_reduction")
                    if vol_reduction is not None:
                        reduction = vol_reduction / current_volatility
                    else:
                        # Apply allocation-based reduction
                        allocation = strategy.get("allocation_percent", 0)
                        reduction = allocation * 0.5  # 50% reduction based on allocation
                    
                    total_reduction += reduction
                
                elif strategy_type == "vix_hedge" or strategy_type == "volatility_hedge":
                    # Volatility hedges primarily target tail risk
                    vol_reduction = strategy.get("expected_impact", {}).get("volatility_reduction")
                    if vol_reduction is not None:
                        reduction = vol_reduction / current_volatility
                    else:
                        # Apply allocation-based reduction
                        allocation = strategy.get("allocation_percent", 0)
                        reduction = allocation * 0.3  # 30% reduction based on allocation
                    
                    total_reduction += reduction
                
                elif strategy_type == "sector_hedge":
                    # Sector hedges reduce volatility for that sector
                    allocation = strategy.get("allocation_percent", 0)
                    # Estimated 30% volatility reduction for the hedged portion
                    reduction = allocation * 0.3
                    
                    total_reduction += reduction
            
            # Apply total reduction (capped)
            capped_reduction = min(total_reduction, max_reduction)
            new_volatility = current_volatility * (1 - capped_reduction)
            
            return max(new_volatility, 1.0)  # Ensure volatility doesn't go below 1%
            
        except Exception as e:
            self.logger.error(f"Error estimating hedged volatility: {e}")
            return current_volatility
    
    def _generate_hedge_exit_conditions(self, risk_analysis: Dict[str, Any], hedge_strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate exit conditions for hedge strategies.
        
        Args:
            risk_analysis: Portfolio risk analysis
            hedge_strategies: List of hedge strategies
            
        Returns:
            List of exit conditions
        """
        try:
            exit_conditions = []
            
            for strategy in hedge_strategies:
                strategy_type = strategy.get("strategy_type")
                
                # General time-based exit
                exit_conditions.append({
                    "strategy_type": strategy_type,
                    "condition_type": "time_based",
                    "parameter": "days_held",
                    "threshold": self.params["max_hedge_duration"],
                    "comparison": "greater_than",
                    "description": f"Exit after {self.params['max_hedge_duration']} days"
                })
                
                # General profit-taking exit
                exit_conditions.append({
                    "strategy_type": strategy_type,
                    "condition_type": "profit_taking",
                    "parameter": "profit_percent",
                    "threshold": self.params["profit_take_threshold"],
                    "comparison": "greater_than",
                    "description": f"Take profit at {self.params['profit_take_threshold']}% gain"
                })
                
                # Strategy-specific exits
                if strategy_type == "index_hedge" or strategy_type == "vix_hedge":
                    # Market condition improvement exit
                    exit_conditions.append({
                        "strategy_type": strategy_type,
                        "condition_type": "market_condition",
                        "parameter": "vix_level",
                        "threshold": self.params["hedge_removal_threshold"],
                        "comparison": "less_than",
                        "description": f"Exit when VIX falls below {self.params['hedge_removal_threshold']}"
                    })
                
                elif strategy_type == "sector_hedge":
                    # Sector performance improvement exit
                    sector = strategy.get("sector")
                    if sector:
                        exit_conditions.append({
                            "strategy_type": strategy_type,
                            "condition_type": "sector_performance",
                            "parameter": f"{sector}_performance",
                            "threshold": 0,
                            "comparison": "greater_than",
                            "description": f"Exit when {sector} sector performance turns positive"
                        })
                
                elif strategy_type == "volatility_hedge":
                    # Volatility condition improvement exit
                    exit_conditions.append({
                        "strategy_type": strategy_type,
                        "condition_type": "volatility_condition",
                        "parameter": "vix_trend",
                        "threshold": "falling",
                        "comparison": "equals",
                        "description": "Exit when VIX trend turns downward"
                    })
            
            return exit_conditions
            
        except Exception as e:
            self.logger.error(f"Error generating hedge exit conditions: {e}")
            return []
    
    def _calculate_hedge_quantity(self, allocation_value: float, current_price: float, symbol: str, strategy_type: str) -> int:
        """
        Calculate quantity for hedge instrument.
        
        Args:
            allocation_value: Allocated value
            current_price: Current price of instrument
            symbol: Instrument symbol
            strategy_type: Type of hedging strategy
            
        Returns:
            Quantity to trade
        """
        try:
            # Get multiplier for futures/options
            multiplier = 1
            
            # For index futures/options, find the appropriate multiplier
            if strategy_type == "index_hedge" or strategy_type == "vix_hedge":
                instrument_info = self._get_instrument_info(symbol)
                if instrument_info:
                    multiplier = instrument_info.get("multiplier", 1)
            
            # Calculate quantity
            quantity = int(allocation_value / (current_price * multiplier))
            
            # Ensure minimum quantity
            return max(quantity, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating hedge quantity: {e}")
            return 0
    
    def _get_market_data(self, symbol: str, exchange: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get market data from database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            days: Number of days to retrieve
            
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
                "timeframe": "day",
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # Get data from database
            cursor = self.db.market_data_collection.find(query).sort("timestamp", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return []
    
    def _get_portfolio_history(self, portfolio_id: str = None) -> List[Dict[str, Any]]:
        """
        Get portfolio value history.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            List of portfolio history data
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
            self.logger.error(f"Error getting portfolio history: {e}")
            return []
    
    def _get_put_call_ratio(self) -> Dict[str, Any]:
        """
        Get put-call ratio data.
        
        Returns:
            Dictionary with put-call ratio data
        """
        try:
            # Query put-call ratio data
            pcr_data = self.db.market_indicators_collection.find_one(
                {"indicator": "put_call_ratio"},
                sort=[("timestamp", -1)]
            )
            
            if not pcr_data:
                return {}
            
            # Get historical PCR data for percentile calculation
            historical_pcr = list(self.db.market_indicators_collection.find(
                {"indicator": "put_call_ratio"}, 
                {"value": 1, "timestamp": 1}
            ).sort("timestamp", -1).limit(30))
            
            pcr_values = [data.get("value", 0) for data in historical_pcr]
            
            # Calculate percentile
            if pcr_values:
                pcr_percentile = sum(1 for x in pcr_values if x < pcr_data.get("value", 0)) / len(pcr_values) * 100
            else:
                pcr_percentile = 50  # Default to 50th percentile
            
            return {
                "put_call_ratio": pcr_data.get("value"),
                "pcr_percentile": pcr_percentile,
                "timestamp": pcr_data.get("timestamp")
            }
            
        except Exception as e:
            self.logger.error(f"Error getting put-call ratio: {e}")
            return {}
    
    def _get_index_hedging_instrument(self) -> Dict[str, Any]:
        """
        Get appropriate index instrument for hedging.
        
        Returns:
            Dictionary with index instrument details
        """
        try:
            # Query index instruments from database
            index_instruments = list(self.db.instruments_collection.find({
                "type": {"$in": ["index_futures", "index_etf", "index_options"]},
                "is_active": True
            }))
            
            if not index_instruments:
                # Default to NIFTY futures if no data available
                return {
                    "symbol": "NIFTY",
                    "name": "NIFTY 50 Futures",
                    "exchange": "NSE",
                    "instrument_type": "index_futures",
                    "expiry": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    "lot_size": 50,
                    "price": 20000,  # Placeholder price
                    "multiplier": 1
                }
            
            # Sort by liquidity or other criteria
            # For simplicity, just take the first one
            return index_instruments[0]
            
        except Exception as e:
            self.logger.error(f"Error getting index hedging instrument: {e}")
            return {}
    
    def _get_vix_hedging_instrument(self) -> Dict[str, Any]:
        """
        Get appropriate VIX instrument for hedging.
        
        Returns:
            Dictionary with VIX instrument details
        """
        try:
            # Query VIX instruments from database
            vix_instruments = list(self.db.instruments_collection.find({
                "symbol": {"$in": ["INDIAVIX", "INDIA VIX"]},
                "is_active": True
            }))
            
            if not vix_instruments:
                # Default to VIX futures if no data available
                return {
                    "symbol": "INDIAVIX",
                    "name": "India VIX Futures",
                    "exchange": "NSE",
                    "instrument_type": "vix_futures",
                    "expiry": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    "lot_size": 75,
                    "price": 15,  # Placeholder price
                    "multiplier": 1
                }
            
            # Return the most appropriate instrument
            return vix_instruments[0]
            
        except Exception as e:
            self.logger.error(f"Error getting VIX hedging instrument: {e}")
            return {}
    
    def _get_sector_hedging_instrument(self, sector: str) -> Dict[str, Any]:
        """
        Get appropriate sector instrument for hedging.
        
        Args:
            sector: Sector name
            
        Returns:
            Dictionary with sector instrument details
        """
        try:
            # Map sector name to sector ETF symbol
            sector_etf_map = {
                "IT": "NIFTIT",
                "Banking": "BANKNIFTY",
                "Pharmaceuticals": "PHARMANIFTY",
                "Financial Services": "FINNIFTY",
                "Auto": "NIFTYAUTO",
                "FMCG": "NIFTYFMCG",
                "Metal": "NIFTYMETAL",
                "Realty": "NIFTYREALTY",
                "Energy": "NIFTYENERGY",
                "PSU Bank": "NIFTYPSUBANK",
                "Media": "NIFTYMEDIA"
            }
            
            # Get ETF symbol
            etf_symbol = sector_etf_map.get(sector)
            
            if not etf_symbol:
                self.logger.warning(f"No ETF mapping found for sector: {sector}")
                return {}
            
            # Query sector ETF from database
            sector_instrument = self.db.instruments_collection.find_one({
                "symbol": etf_symbol,
                "is_active": True
            })
            
            if not sector_instrument:
                # Default to sector futures if no data available
                return {
                    "symbol": etf_symbol,
                    "name": f"{sector} Index Futures",
                    "exchange": "NSE",
                    "instrument_type": "sector_futures",
                    "expiry": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    "lot_size": 25,
                    "price": 10000,  # Placeholder price
                    "multiplier": 1,
                    "sector": sector
                }
            
            # Add sector information
            sector_instrument["sector"] = sector
            
            return sector_instrument
            
        except Exception as e:
            self.logger.error(f"Error getting sector hedging instrument for {sector}: {e}")
            return {}
    
    def _get_volatility_hedging_instrument(self) -> Dict[str, Any]:
        """
        Get appropriate volatility instrument for hedging.
        
        Returns:
            Dictionary with volatility instrument details
        """
        try:
            # Since VIX options or direct volatility ETFs may not be readily available in India,
            # we can use NIFTY Put options as a proxy for volatility exposure
            
            # Query NIFTY Put options from database
            nifty_puts = list(self.db.instruments_collection.find({
                "symbol": "NIFTY",
                "instrument_type": "index_options",
                "option_type": "PE",
                "is_active": True
            }))
            
            if not nifty_puts:
                # Current NIFTY price (placeholder)
                nifty_price = 20000
                
                # Calculate appropriate strike price (5% OTM)
                strike_price = math.floor(nifty_price * 0.95 / 100) * 100
                
                # Default to NIFTY Put option if no data available
                return {
                    "symbol": "NIFTY",
                    "name": f"NIFTY 50 {strike_price} PE",
                    "exchange": "NSE",
                    "instrument_type": "index_options",
                    "option_type": "PE",
                    "strike_price": strike_price,
                    "expiry": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    "lot_size": 50,
                    "price": 300,  # Placeholder price
                    "multiplier": 1
                }
            
            # Find an appropriate strike (about 5% OTM)
            for put in nifty_puts:
                if put.get("option_type") == "PE" and put.get("days_to_expiry", 0) > 20:
                    return put
            
            # If no suitable put found, return the first one
            return nifty_puts[0]
            
        except Exception as e:
            self.logger.error(f"Error getting volatility hedging instrument: {e}")
            return {}
    
    def _get_instrument_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get instrument information from database.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Dictionary with instrument details
        """
        try:
            # Query instrument from database
            instrument = self.db.instruments_collection.find_one({
                "symbol": symbol,
                "is_active": True
            })
            
            return instrument or {}
            
        except Exception as e:
            self.logger.error(f"Error getting instrument info for {symbol}: {e}")
            return {}
    
    def _get_symbol_volatility(self, symbol: str, exchange: str = "NSE") -> float:
        """
        Calculate symbol volatility.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Annualized volatility percentage
        """
        try:
            # Get historical data
            data = self._get_market_data(symbol, exchange, days=30)
            
            if not data or len(data) < 5:
                return 0.0
            
            # Calculate returns
            prices = [item["close"] for item in data]
            returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
            
            # Calculate volatility (annualized)
            daily_volatility = np.std(returns)
            annualized_volatility = daily_volatility * np.sqrt(252) * 100
            
            return annualized_volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.0
    
    def _get_sector_performance(self, sector: str, exchange: str = "NSE") -> float:
        """
        Get sector performance over past month.
        
        Args:
            sector: Sector name
            exchange: Stock exchange
            
        Returns:
            Sector performance percentage
        """
        try:
            # Map sector to index
            sector_index_map = {
                "IT": "NIFTIT",
                "Banking": "BANKNIFTY",
                "Pharmaceuticals": "PHARMANIFTY",
                "Financial Services": "FINNIFTY",
                "Auto": "NIFTYAUTO",
                "FMCG": "NIFTYFMCG",
                "Metal": "NIFTYMETAL",
                "Realty": "NIFTYREALTY",
                "Energy": "NIFTYENERGY",
                "PSU Bank": "NIFTYPSUBANK",
                "Media": "NIFTYMEDIA"
            }
            
            index_symbol = sector_index_map.get(sector, None)
            
            if not index_symbol:
                return 0.0
            
            # Get historical data
            data = self._get_market_data(index_symbol, exchange, days=30)
            
            if not data or len(data) < 5:
                return 0.0
            
            # Calculate performance
            current_value = data[-1]["close"]
            past_value = data[0]["close"]
            
            performance = (current_value / past_value - 1) * 100
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating sector performance for {sector}: {e}")
            return 0.0
    
    def _get_current_hedges(self) -> List[Dict[str, Any]]:
        """
        Get current hedge positions.
        
        Returns:
            List of current hedge positions
        """
        try:
            # Query hedge positions from database
            hedges = list(self.db.positions_collection.find({
                "is_hedge": True,
                "status": "active"
            }))
            
            return hedges
            
        except Exception as e:
            self.logger.error(f"Error getting current hedges: {e}")
            return []
    
    def _calculate_hedge_adjustments(self, recommended_hedges: List[Dict[str, Any]], current_hedges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate required adjustments to hedging positions.
        
        Args:
            recommended_hedges: Recommended hedge instruments
            current_hedges: Current hedge positions
            
        Returns:
            Dictionary with hedge adjustments
        """
        try:
            # Map current hedges by symbol
            current_hedge_map = {hedge.get("symbol"): hedge for hedge in current_hedges}
            
            # Track adjustments
            new_hedges = []
            adjust_hedges = []
            remove_hedges = []
            
            # Check for hedges to add or adjust
            for hedge in recommended_hedges:
                symbol = hedge.get("symbol")
                allocation_value = hedge.get("allocation_value", 0)
                
                # Skip non-actionable hedges
                if not symbol or allocation_value <= 0:
                    continue
                
                if symbol in current_hedge_map:
                    # Existing hedge that needs adjustment
                    current_hedge = current_hedge_map[symbol]
                    current_value = current_hedge.get("position_value", 0)
                    
                    # Check if adjustment is significant (>10% change)
                    if abs(allocation_value - current_value) / current_value > 0.1:
                        adjust_hedges.append({
                            "symbol": symbol,
                            "current_value": current_value,
                            "target_value": allocation_value,
                            "adjustment": allocation_value - current_value,
                            "hedge_details": hedge
                        })
                else:
                    # New hedge to add
                    new_hedges.append({
                        "symbol": symbol,
                        "allocation_value": allocation_value,
                        "hedge_details": hedge
                    })
            
            # Check for hedges to remove
            for symbol, hedge in current_hedge_map.items():
                # If not in recommended hedges, remove it
                if not any(h.get("symbol") == symbol for h in recommended_hedges):
                    remove_hedges.append({
                        "symbol": symbol,
                        "position_value": hedge.get("position_value", 0),
                        "hedge_details": hedge
                    })
            
            return {
                "new_hedges": new_hedges,
                "adjust_hedges": adjust_hedges,
                "remove_hedges": remove_hedges
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating hedge adjustments: {e}")
            return {"new_hedges": [], "adjust_hedges": [], "remove_hedges": []}
    
    def _execute_hedge_adjustments(self, hedge_adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute hedge adjustments.
        
        Args:
            hedge_adjustments: Dictionary with hedge adjustments
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Track execution results
            execution_results = {
                "new_hedges": [],
                "adjusted_hedges": [],
                "removed_hedges": [],
                "errors": []
            }
            
            # Execute new hedges
            for hedge in hedge_adjustments.get("new_hedges", []):
                try:
                    # Execute new hedge
                    hedge_instrument = hedge.get("hedge_details", {}).get("instrument", {})
                    symbol = hedge_instrument.get("symbol")
                    instrument_type = hedge_instrument.get("instrument_type")
                    price = hedge_instrument.get("price", 0)
                    
                    # Calculate quantity
                    quantity = self._calculate_hedge_quantity(
                        hedge.get("allocation_value", 0),
                        price,
                        symbol,
                        hedge.get("hedge_details", {}).get("strategy_type", "")
                    )
                    
                    if quantity <= 0:
                        raise ValueError(f"Invalid quantity calculated for {symbol}")
                    
                    # Execute order (placeholder for actual execution logic)
                    execution_result = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "price": price,
                        "order_type": "market",
                        "status": "executed",
                        "strategy_type": hedge.get("hedge_details", {}).get("strategy_type", ""),
                        "timestamp": datetime.now()
                    }
                    
                    execution_results["new_hedges"].append(execution_result)
                    
                except Exception as e:
                    execution_results["errors"].append({
                        "hedge": hedge,
                        "error": str(e),
                        "type": "new_hedge_error"
                    })
            
            # Execute hedge adjustments
            for hedge in hedge_adjustments.get("adjust_hedges", []):
                try:
                    # Execute hedge adjustment
                    symbol = hedge.get("symbol")
                    adjustment = hedge.get("adjustment", 0)
                    
                    if adjustment == 0:
                        continue
                    
                    hedge_instrument = hedge.get("hedge_details", {}).get("instrument", {})
                    price = hedge_instrument.get("price", 0)
                    
                    # Calculate adjustment quantity
                    quantity = int(abs(adjustment) / price) if price > 0 else 0
                    
                    if quantity <= 0:
                        continue
                    
                    # Determine action
                    action = "buy" if adjustment > 0 else "sell"
                    
                    # Execute order (placeholder for actual execution logic)
                    execution_result = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "price": price,
                        "order_type": "market",
                        "action": action,
                        "status": "executed",
                        "strategy_type": hedge.get("hedge_details", {}).get("strategy_type", ""),
                        "timestamp": datetime.now()
                    }
                    
                    execution_results["adjusted_hedges"].append(execution_result)
                    
                except Exception as e:
                    execution_results["errors"].append({
                        "hedge": hedge,
                        "error": str(e),
                        "type": "adjust_hedge_error"
                    })
            
            # Execute hedge removals
            for hedge in hedge_adjustments.get("remove_hedges", []):
                try:
                    # Execute hedge removal
                    symbol = hedge.get("symbol")
                    
                    # Execute order (placeholder for actual execution logic)
                    execution_result = {
                        "symbol": symbol,
                        "quantity": hedge.get("hedge_details", {}).get("quantity", 0),
                        "price": hedge.get("hedge_details", {}).get("current_price", 0),
                        "order_type": "market",
                        "action": "sell",
                        "status": "executed",
                        "strategy_type": hedge.get("hedge_details", {}).get("strategy_type", ""),
                        "timestamp": datetime.now()
                    }
                    
                    execution_results["removed_hedges"].append(execution_result)
                    
                except Exception as e:
                    execution_results["errors"].append({
                        "hedge": hedge,
                        "error": str(e),
                        "type": "remove_hedge_error"
                    })
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Error executing hedge adjustments: {e}")
            return {"new_hedges": [], "adjusted_hedges": [], "removed_hedges": [], "errors": [str(e)]}