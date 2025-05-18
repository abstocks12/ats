# futures_strategies.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

class FuturesStrategies:
    def __init__(self, db_connector):
        """Initialize the futures strategies module"""
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, symbol, exchange, timeframe="day"):
        """
        Analyze futures for potential trading opportunities
        Returns a dictionary with analysis results and trading signals
        """
        try:
            # Get historical futures data
            futures_data = self._get_futures_data(symbol, exchange, timeframe)
            if futures_data is None or len(futures_data) < 20:
                return {"status": "error", "message": "Insufficient data for analysis"}
            
            # Get spot data for basis analysis
            spot_data = self._get_spot_data(symbol, exchange, timeframe)
            
            # Calculate basis (difference between futures and spot)
            basis_analysis = self._analyze_basis(futures_data, spot_data)
            
            # Analyze futures specific patterns
            momentum_signals = self._analyze_momentum(futures_data)
            term_structure = self._analyze_term_structure(symbol, exchange)
            roll_opportunities = self._analyze_roll_opportunities(symbol, exchange)
            volume_profile = self._analyze_volume_profile(futures_data)
            
            # Consolidate analysis
            analysis_result = {
                "symbol": symbol,
                "exchange": exchange,
                "futures_analysis": {
                    "timeframe": timeframe,
                    "timestamp": datetime.now(),
                    "basis_analysis": basis_analysis,
                    "momentum_signals": momentum_signals,
                    "term_structure": term_structure,
                    "roll_opportunities": roll_opportunities,
                    "volume_profile": volume_profile
                }
            }
            
            # Generate trading signals
            signals = self._generate_signals(analysis_result)
            analysis_result["signals"] = signals
            
            # Save analysis to database
            self._save_analysis(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing futures for {symbol}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_futures_data(self, symbol, exchange, timeframe):
        """Retrieve historical futures data"""
        try:
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": "futures",
                "timeframe": timeframe
            }
            
            # Get the last 100 data points
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            ).sort("timestamp", -1).limit(100))
            
            if not data:
                return None
                
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting futures data: {str(e)}")
            return None
    
    def _get_spot_data(self, symbol, exchange, timeframe):
        """Retrieve historical spot data for the underlying"""
        try:
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": "equity",
                "timeframe": timeframe
            }
            
            # Get the last 100 data points
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "close": 1}
            ).sort("timestamp", -1).limit(100))
            
            if not data:
                return None
                
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting spot data: {str(e)}")
            return None
    
    def _analyze_basis(self, futures_data, spot_data):
        """
        Analyze the basis (futures price - spot price)
        This can indicate market sentiment and hedging opportunities
        """
        if spot_data is None or futures_data is None:
            return {"status": "error", "message": "Missing data for basis analysis"}
        
        try:
            # Merge dataframes on timestamp
            merged_data = pd.merge(
                futures_data[["timestamp", "close"]], 
                spot_data[["timestamp", "close"]], 
                on="timestamp", 
                suffixes=("_futures", "_spot")
            )
            
            if len(merged_data) < 5:
                return {"status": "error", "message": "Insufficient overlapping data points"}
            
            # Calculate basis and basis percentage
            merged_data["basis"] = merged_data["close_futures"] - merged_data["close_spot"]
            merged_data["basis_percent"] = (merged_data["basis"] / merged_data["close_spot"]) * 100
            
            # Calculate statistics
            current_basis = merged_data["basis"].iloc[-1]
            current_basis_percent = merged_data["basis_percent"].iloc[-1]
            mean_basis = merged_data["basis"].mean()
            std_basis = merged_data["basis"].std()
            z_score = (current_basis - mean_basis) / std_basis if std_basis > 0 else 0
            
            # Determine if basis is widening or narrowing
            basis_5d_ago = merged_data["basis"].iloc[-6] if len(merged_data) > 5 else merged_data["basis"].iloc[0]
            basis_change = current_basis - basis_5d_ago
            basis_change_percent = (basis_change / abs(basis_5d_ago)) * 100 if basis_5d_ago != 0 else 0
            
            # Interpret basis
            if z_score > 2:
                interpretation = "Unusually wide basis - potential for convergence or roll-down"
            elif z_score < -2:
                interpretation = "Unusually narrow basis - potential for divergence or roll-up"
            else:
                interpretation = "Basis within normal range"
            
            return {
                "current_basis": current_basis,
                "current_basis_percent": current_basis_percent,
                "mean_basis": mean_basis,
                "basis_z_score": z_score,
                "basis_change_5d": basis_change,
                "basis_change_percent_5d": basis_change_percent,
                "basis_widening": basis_change > 0,
                "interpretation": interpretation
            }
            
        except Exception as e:
            self.logger.error(f"Error in basis analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_momentum(self, futures_data):
        """Analyze futures momentum characteristics"""
        try:
            df = futures_data.copy()
            
            # Calculate momentum indicators
            # 5-day rate of change
            df["roc_5"] = (df["close"] / df["close"].shift(5) - 1) * 100
            
            # 20-day rate of change
            df["roc_20"] = (df["close"] / df["close"].shift(20) - 1) * 100
            
            # ADX for trend strength
            df = self._calculate_adx(df, 14)
            
            # Get current values
            current_roc_5 = df["roc_5"].iloc[-1] if "roc_5" in df else None
            current_roc_20 = df["roc_20"].iloc[-1] if "roc_20" in df else None
            current_adx = df["adx"].iloc[-1] if "adx" in df else None
            
            # Determine trend strength
            trend_strength = "No trend"
            if current_adx:
                if current_adx < 20:
                    trend_strength = "Weak/Absent"
                elif current_adx < 30:
                    trend_strength = "Moderate"
                else:
                    trend_strength = "Strong"
            
            # Determine trend direction
            trend_direction = "Neutral"
            if current_roc_5 and current_roc_20:
                if current_roc_5 > 1 and current_roc_20 > 0:
                    trend_direction = "Bullish"
                elif current_roc_5 < -1 and current_roc_20 < 0:
                    trend_direction = "Bearish"
            
            return {
                "roc_5d": current_roc_5,
                "roc_20d": current_roc_20,
                "adx": current_adx,
                "trend_strength": trend_strength,
                "trend_direction": trend_direction
            }
            
        except Exception as e:
            self.logger.error(f"Error in momentum analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index (ADX)"""
        try:
            # Calculate True Range
            df["tr1"] = abs(df["high"] - df["low"])
            df["tr2"] = abs(df["high"] - df["close"].shift(1))
            df["tr3"] = abs(df["low"] - df["close"].shift(1))
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            
            # Calculate +DM and -DM
            df["plus_dm"] = 0.0
            df["minus_dm"] = 0.0
            
            for i in range(1, len(df)):
                # +DM
                if df["high"].iloc[i] > df["high"].iloc[i-1]:
                    df.at[df.index[i], "plus_dm"] = max(df["high"].iloc[i] - df["high"].iloc[i-1], 0)
                else:
                    df.at[df.index[i], "plus_dm"] = 0
                
                # -DM
                if df["low"].iloc[i-1] > df["low"].iloc[i]:
                    df.at[df.index[i], "minus_dm"] = max(df["low"].iloc[i-1] - df["low"].iloc[i], 0)
                else:
                    df.at[df.index[i], "minus_dm"] = 0
                
                # If +DM and -DM are both 0, set both to 0
                if df.at[df.index[i], "plus_dm"] == df.at[df.index[i], "minus_dm"]:
                    df.at[df.index[i], "plus_dm"] = 0
                    df.at[df.index[i], "minus_dm"] = 0
            
            # Calculate smoothed values
            df["tr_" + str(period)] = df["tr"].rolling(window=period).sum()
            df["plus_dm_" + str(period)] = df["plus_dm"].rolling(window=period).sum()
            df["minus_dm_" + str(period)] = df["minus_dm"].rolling(window=period).sum()
            
            # Calculate +DI and -DI
            df["plus_di_" + str(period)] = 100 * (df["plus_dm_" + str(period)] / df["tr_" + str(period)])
            df["minus_di_" + str(period)] = 100 * (df["minus_dm_" + str(period)] / df["tr_" + str(period)])
            
            # Calculate DX
            df["dx_" + str(period)] = 100 * abs(df["plus_di_" + str(period)] - df["minus_di_" + str(period)]) / (df["plus_di_" + str(period)] + df["minus_di_" + str(period)])
            
            # Calculate ADX
            df["adx"] = df["dx_" + str(period)].rolling(window=period).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {str(e)}")
            return df
    
    def _analyze_term_structure(self, symbol, exchange):
        """
        Analyze the futures term structure (near, mid, far contracts)
        This can indicate market sentiment and calendar spread opportunities
        """
        try:
            # Get near, mid, and far month futures
            contracts = self._get_futures_contracts(symbol, exchange)
            
            if not contracts or len(contracts) < 2:
                return {"status": "error", "message": "Insufficient futures contracts"}
            
            # Get close prices for each contract
            near_price = contracts[0].get("close")
            mid_price = contracts[1].get("close") if len(contracts) > 1 else None
            far_price = contracts[2].get("close") if len(contracts) > 2 else None
            
            # Calculate term structure slopes
            near_mid_slope = ((mid_price / near_price) - 1) * 100 if near_price and mid_price else None
            mid_far_slope = ((far_price / mid_price) - 1) * 100 if mid_price and far_price else None
            
            # Determine term structure shape
            shape = "Unknown"
            if near_mid_slope is not None and mid_far_slope is not None:
                if near_mid_slope > 0 and mid_far_slope > 0:
                    shape = "Contango (upward sloping)"
                elif near_mid_slope < 0 and mid_far_slope < 0:
                    shape = "Backwardation (downward sloping)"
                elif near_mid_slope > 0 and mid_far_slope < 0:
                    shape = "Humped"
                elif near_mid_slope < 0 and mid_far_slope > 0:
                    shape = "Inverted humped"
            elif near_mid_slope is not None:
                if near_mid_slope > 0:
                    shape = "Contango (upward sloping)"
                else:
                    shape = "Backwardation (downward sloping)"
            
            # Market interpretation
            interpretation = "Neutral"
            if shape == "Contango (upward sloping)":
                interpretation = "Bearish - market expects higher future prices"
            elif shape == "Backwardation (downward sloping)":
                interpretation = "Bullish - market expects lower future prices"
            
            return {
                "near_month": contracts[0].get("expiry"),
                "mid_month": contracts[1].get("expiry") if len(contracts) > 1 else None,
                "far_month": contracts[2].get("expiry") if len(contracts) > 2 else None,
                "near_price": near_price,
                "mid_price": mid_price,
                "far_price": far_price,
                "near_mid_slope": near_mid_slope,
                "mid_far_slope": mid_far_slope,
                "term_structure": shape,
                "interpretation": interpretation
            }
            
        except Exception as e:
            self.logger.error(f"Error in term structure analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_futures_contracts(self, symbol, exchange):
        """Get all active futures contracts for a symbol"""
        try:
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "instrument_type": "futures",
                "status": "active"
            }
            
            # Find contracts and sort by expiry
            contracts = list(self.db.instrument_collection.find(query).sort("expiry", 1))
            
            # Get latest prices for each contract
            for contract in contracts:
                latest_price = self.db.market_data_collection.find_one(
                    {
                        "symbol": symbol,
                        "exchange": exchange,
                        "instrument_type": "futures",
                        "contract_id": contract["_id"]
                    },
                    {"close": 1}
                )
                
                if latest_price:
                    contract["close"] = latest_price["close"]
            
            return contracts
            
        except Exception as e:
            self.logger.error(f"Error getting futures contracts: {str(e)}")
            return []
    
    def _analyze_roll_opportunities(self, symbol, exchange):
        """
        Analyze rollover opportunities near expiry
        """
        try:
            contracts = self._get_futures_contracts(symbol, exchange)
            
            if not contracts or len(contracts) < 2:
                return {"status": "error", "message": "Insufficient futures contracts"}
            
            # Get near contract
            near_contract = contracts[0]
            
            # Calculate days to expiry
            expiry_date = near_contract.get("expiry")
            if not expiry_date:
                return {"status": "error", "message": "Missing expiry information"}
            
            days_to_expiry = (expiry_date - datetime.now()).days
            
            # Get prices
            near_price = near_contract.get("close")
            mid_price = contracts[1].get("close") if len(contracts) > 1 else None
            
            if not near_price or not mid_price:
                return {"status": "error", "message": "Missing price information"}
            
            # Calculate roll cost
            roll_cost = mid_price - near_price
            roll_cost_percent = (roll_cost / near_price) * 100
            
            # Calculate daily roll cost
            daily_roll_cost = roll_cost_percent / days_to_expiry if days_to_expiry > 0 else None
            
            # Get historical roll costs
            historical_rolls = self._get_historical_roll_costs(symbol, exchange)
            
            # Compare with historical average
            avg_roll_cost = sum([r["roll_cost_percent"] for r in historical_rolls]) / len(historical_rolls) if historical_rolls else None
            roll_vs_avg = (roll_cost_percent / avg_roll_cost) if avg_roll_cost else None
            
            # Strategy recommendation
            strategy = "None"
            if days_to_expiry <= 5:
                strategy = "Roll over positions to next month"
            elif days_to_expiry <= 10:
                if roll_cost_percent > 0.5:
                    strategy = "Consider early rollover to avoid high roll costs"
                else:
                    strategy = "Monitor roll cost daily"
            
            return {
                "near_contract_expiry": expiry_date,
                "days_to_expiry": days_to_expiry,
                "roll_cost": roll_cost,
                "roll_cost_percent": roll_cost_percent,
                "daily_roll_cost": daily_roll_cost,
                "average_historical_roll_cost": avg_roll_cost,
                "roll_vs_historical_avg": roll_vs_avg,
                "roll_strategy": strategy
            }
            
        except Exception as e:
            self.logger.error(f"Error in roll opportunity analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_historical_roll_costs(self, symbol, exchange):
        """Get historical roll costs for a symbol"""
        try:
            # Query for historical roll records
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "data_type": "roll_cost"
            }
            
            # Get last 6 roll records
            rolls = list(self.db.market_metadata_collection.find(query).sort("date", -1).limit(6))
            
            return rolls
            
        except Exception as e:
            self.logger.error(f"Error getting historical roll costs: {str(e)}")
            return []
    
    def _analyze_volume_profile(self, futures_data):
        """
        Analyze the volume profile of a futures contract
        """
        try:
            df = futures_data.copy()
            
            if "volume" not in df.columns or len(df) < 5:
                return {"status": "error", "message": "Insufficient volume data"}
            
            # Calculate volume metrics
            avg_volume_5d = df["volume"].tail(5).mean()
            avg_volume_20d = df["volume"].tail(20).mean() if len(df) >= 20 else None
            
            # Calculate volume trend
            volume_change = (avg_volume_5d / avg_volume_20d - 1) * 100 if avg_volume_20d else None
            
            # Calculate relative volume
            current_volume = df["volume"].iloc[-1]
            relative_volume = current_volume / avg_volume_20d if avg_volume_20d else None
            
            # Classify volume profile
            profile = "Normal"
            if relative_volume:
                if relative_volume > 2.0:
                    profile = "Extremely High"
                elif relative_volume > 1.5:
                    profile = "High"
                elif relative_volume < 0.5:
                    profile = "Low"
                elif relative_volume < 0.25:
                    profile = "Extremely Low"
            
            # Interpretation
            interpretation = "Neutral"
            if profile == "Extremely High":
                interpretation = "Significant market interest - potential trend development"
            elif profile == "High":
                interpretation = "Increased market interest - validate with price action"
            elif profile == "Low":
                interpretation = "Reduced market interest - potential consolidation"
            elif profile == "Extremely Low":
                interpretation = "Very low interest - await improved liquidity"
            
            return {
                "current_volume": current_volume,
                "avg_volume_5d": avg_volume_5d,
                "avg_volume_20d": avg_volume_20d,
                "volume_change_percent": volume_change,
                "relative_volume": relative_volume,
                "volume_profile": profile,
                "interpretation": interpretation
            }
            
        except Exception as e:
            self.logger.error(f"Error in volume profile analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _generate_signals(self, analysis_result):
        """
        Generate trading signals based on the analysis
        """
        try:
            signals = []
            
            # Get components of the analysis
            basis = analysis_result["futures_analysis"]["basis_analysis"]
            momentum = analysis_result["futures_analysis"]["momentum_signals"]
            term_structure = analysis_result["futures_analysis"]["term_structure"]
            roll = analysis_result["futures_analysis"]["roll_opportunities"]
            volume = analysis_result["futures_analysis"]["volume_profile"]
            
            # Basis arbitrage signal
            if "basis_z_score" in basis and abs(basis["basis_z_score"]) > 2:
                if basis["basis_z_score"] > 2:
                    signals.append({
                        "strategy": "Basis Arbitrage",
                        "signal": "Short Futures, Long Spot",
                        "strength": min(abs(basis["basis_z_score"]) / 2, 5),
                        "reason": "Unusually wide basis suggests futures overvalued relative to spot"
                    })
                else:
                    signals.append({
                        "strategy": "Basis Arbitrage",
                        "signal": "Long Futures, Short Spot",
                        "strength": min(abs(basis["basis_z_score"]) / 2, 5),
                        "reason": "Unusually narrow basis suggests futures undervalued relative to spot"
                    })
            
            # Momentum signal
            if "trend_direction" in momentum and "trend_strength" in momentum:
                if momentum["trend_direction"] == "Bullish" and momentum["trend_strength"] in ["Moderate", "Strong"]:
                    signals.append({
                        "strategy": "Momentum",
                        "signal": "Long Futures",
                        "strength": 4 if momentum["trend_strength"] == "Strong" else 3,
                        "reason": f"{momentum['trend_strength']} bullish trend with {momentum['roc_5d']:.2f}% 5-day ROC"
                    })
                elif momentum["trend_direction"] == "Bearish" and momentum["trend_strength"] in ["Moderate", "Strong"]:
                    signals.append({
                        "strategy": "Momentum",
                        "signal": "Short Futures",
                        "strength": 4 if momentum["trend_strength"] == "Strong" else 3,
                        "reason": f"{momentum['trend_strength']} bearish trend with {momentum['roc_5d']:.2f}% 5-day ROC"
                    })
            
            # Calendar spread based on term structure
            if "term_structure" in term_structure and term_structure["term_structure"] != "Unknown":
                if term_structure["term_structure"] == "Contango (upward sloping)":
                    signals.append({
                        "strategy": "Calendar Spread",
                        "signal": "Short Near Month, Long Far Month",
                        "strength": 3,
                        "reason": "Contango term structure suggests shorting near month contracts"
                    })
                elif term_structure["term_structure"] == "Backwardation (downward sloping)":
                    signals.append({
                        "strategy": "Calendar Spread",
                        "signal": "Long Near Month, Short Far Month",
                        "strength": 3,
                        "reason": "Backwardation term structure suggests longing near month contracts"
                    })
            
            # Roll strategy
            if isinstance(roll, dict) and "roll_strategy" in roll and roll["roll_strategy"] != "None":
                signals.append({
                    "strategy": "Roll Management",
                    "signal": roll["roll_strategy"],
                    "strength": 5 if roll.get("days_to_expiry", 100) <= 5 else 3,
                    "reason": f"{roll.get('days_to_expiry')} days to expiry with {roll.get('roll_cost_percent', 0):.2f}% roll cost"
                })
            
            # Volume-based signal
            if isinstance(volume, dict) and "volume_profile" in volume and "interpretation" in volume:
                if volume["volume_profile"] in ["High", "Extremely High"] and momentum["trend_direction"] != "Neutral":
                    signals.append({
                        "strategy": "Volume Confirmation",
                        "signal": "Long Futures" if momentum["trend_direction"] == "Bullish" else "Short Futures",
                        "strength": 2,
                        "reason": f"{volume['volume_profile']} volume confirms {momentum['trend_direction'].lower()} trend"
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return []
    
    def _save_analysis(self, analysis_result):
        """Save analysis results to database"""
        try:
            # Create document for database
            doc = {
                "symbol": analysis_result["symbol"],
                "exchange": analysis_result["exchange"],
                "analysis_type": "futures",
                "timestamp": datetime.now(),
                "data": analysis_result
            }
            
            # Insert or update in database
            self.db.analysis_collection.replace_one(
                {
                    "symbol": analysis_result["symbol"],
                    "exchange": analysis_result["exchange"],
                    "analysis_type": "futures"
                },
                doc,
                upsert=True
            )
            
        except Exception as e:
            self.logger.error(f"Error saving analysis: {str(e)}")