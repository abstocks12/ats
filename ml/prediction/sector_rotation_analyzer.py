# ml/prediction/sector_rotation_analyzer.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import linregress

class SectorRotationAnalyzer:
    """Analyze and predict sector rotation patterns."""
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the sector rotation analyzer.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'analysis_periods': [30, 90, 180],  # Days for trend analysis
            'num_clusters': 4,  # Number of sector clusters
            'min_sector_symbols': 3,  # Minimum symbols per sector for analysis
            'default_sectors': [
                'Technology', 'Financial', 'Healthcare', 'Consumer', 'Industrial',
                'Energy', 'Materials', 'Utilities', 'Communication', 'Real Estate'
            ],
            'rotation_stages': [
                'Early Expansion', 'Late Expansion', 
                'Early Contraction', 'Late Contraction'
            ]
        }
        
        # Current market cycle stage
        self.current_stage = None
        
        # Sector performance data
        self.sector_data = None
    
    def set_config(self, config):
        """
        Set analyzer configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated sector rotation analyzer configuration: {self.config}")
    
    def get_sector_data(self, start_date=None, end_date=None):
        """
        Get sector performance data.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Sector data by period
        """
        # Set default date range
        end_date = end_date or datetime.now()
        
        # Calculate start date based on longest analysis period
        max_period = max(self.config['analysis_periods'])
        start_date = start_date or (end_date - timedelta(days=max_period * 1.5))
        
        self.logger.info(f"Getting sector data from {start_date} to {end_date}")
        
        # Get all active symbols with sector information
        cursor = self.db.portfolio_collection.find({
            'status': 'active',
            'sector': {'$exists': True}
        })
        
        symbols_by_sector = {}
        
        for doc in cursor:
            sector = doc.get('sector')
            symbol = doc.get('symbol')
            exchange = doc.get('exchange')
            
            if not all([sector, symbol, exchange]):
                continue
                
            if sector not in symbols_by_sector:
                symbols_by_sector[sector] = []
                
            symbols_by_sector[sector].append((symbol, exchange))
        
        # Filter sectors with too few symbols
        min_symbols = self.config['min_sector_symbols']
        symbols_by_sector = {
            sector: symbols for sector, symbols in symbols_by_sector.items()
            if len(symbols) >= min_symbols
        }
        
        if not symbols_by_sector:
            # Use default sectors if no sector data available
            self.logger.warning("No sector data available, using default sectors")
            return self._generate_default_sector_data(start_date, end_date)
        
        # Calculate performance for each analysis period
        performance_by_period = {}
        
        for period in self.config['analysis_periods']:
            period_start = end_date - timedelta(days=period)
            
            sector_performance = {}
            
            for sector, symbols in symbols_by_sector.items():
                # Calculate average performance for symbols in this sector
                symbol_performances = []
                
                for symbol, exchange in symbols:
                    perf = self._calculate_symbol_performance(
                        symbol, exchange, period_start, end_date
                    )
                    
                    if perf is not None:
                        symbol_performances.append(perf)
                
                if symbol_performances:
                    avg_performance = np.mean(symbol_performances)
                    
                    sector_performance[sector] = {
                        'performance': avg_performance,
                        'symbols_count': len(symbol_performances),
                        'symbols_used': len(symbols)
                    }
            
            # Sort sectors by performance
            sorted_sectors = sorted(
                sector_performance.items(),
                key=lambda x: x[1]['performance'],
                reverse=True
            )
            
            performance_by_period[period] = {
                'start_date': period_start,
                'end_date': end_date,
                'sectors': dict(sorted_sectors)
            }
        
        self.sector_data = performance_by_period
        
        self.logger.info(f"Collected performance data for {len(symbols_by_sector)} sectors")
        
        return performance_by_period
    
    def _calculate_symbol_performance(self, symbol, exchange, start_date, end_date):
        """
        Calculate performance for a symbol over a period.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            float: Performance percentage or None if data not available
        """
        try:
            # Get market data
            query = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': 'day',
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            # Get first and last data points
            start_data = self.db.market_data_collection.find_one(
                query, sort=[('timestamp', 1)]
            )
            
            end_data = self.db.market_data_collection.find_one(
                query, sort=[('timestamp', -1)]
            )
            
            if not start_data or not end_data:
                return None
                
            start_close = start_data.get('close')
            end_close = end_data.get('close')
            
            if not start_close or not end_close:
                return None
                
            # Calculate performance
            performance = (end_close - start_close) / start_close
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"Error calculating performance for {symbol}: {e}")
            return None
    
    def _generate_default_sector_data(self, start_date, end_date):
        """
        Generate default sector data if real data not available.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Generated sector data
        """
        # Get market index data to generate realistic sector performance
        try:
            index_symbol = '^NSEI'  # Nifty 50 for Indian market
            
            query = {
                'symbol': index_symbol,
                'timeframe': 'day',
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            cursor = self.db.market_data_collection.find(query).sort('timestamp', 1)
            index_data = list(cursor)
            
            if not index_data:
                # Fallback to random data
                return self._generate_random_sector_data(start_date, end_date)
                
            # Calculate index performance for reference
            index_closes = [data.get('close', 0) for data in index_data]
            index_performance = (index_closes[-1] - index_closes[0]) / index_closes[0]
            
            # Generate sector performances based on typical sector rotation model
            # and current market conditions
            performance_by_period = {}
            
            for period in self.config['analysis_periods']:
                period_start = end_date - timedelta(days=period)
                
                # Filter data for this period
                period_data = [d for d in index_data if d['timestamp'] >= period_start]
                
                if not period_data:
                    continue
                    
                period_closes = [data.get('close', 0) for data in period_data]
                period_performance = (period_closes[-1] - period_closes[0]) / period_closes[0]
                
                # Determine market cycle based on performance
                is_bull = period_performance > 0
                
                # Generate sector performances based on typical rotation
                sector_performance = {}
                
                if is_bull:
                    # Bull market sector rotation
                    sector_performance = {
                        'Technology': {'performance': period_performance * 1.3},
                        'Financial': {'performance': period_performance * 1.2},
                        'Consumer': {'performance': period_performance * 1.1},
                        'Industrial': {'performance': period_performance * 1.05},
                        'Healthcare': {'performance': period_performance * 0.9},
                        'Materials': {'performance': period_performance * 0.95},
                        'Energy': {'performance': period_performance * 0.9},
                        'Utilities': {'performance': period_performance * 0.7},
                        'Real Estate': {'performance': period_performance * 1.0},
                        'Communication': {'performance': period_performance * 1.1}
                    }
                else:
                    # Bear market sector rotation
                    sector_performance = {
                        'Technology': {'performance': period_performance * 1.3},
                        'Financial': {'performance': period_performance * 1.2},
                        'Consumer': {'performance': period_performance * 0.8},
                        'Industrial': {'performance': period_performance * 1.1},
                        'Healthcare': {'performance': period_performance * 0.7},
                        'Materials': {'performance': period_performance * 1.0},
                        'Energy': {'performance': period_performance * 0.9},
                        'Utilities': {'performance': period_performance * 0.7},
                        'Real Estate': {'performance': period_performance * 1.3},
                        'Communication': {'performance': period_performance * 1.0}
                    }
                
                # Add dummy counts
                for sector in sector_performance:
                    sector_performance[sector]['symbols_count'] = 5
                    sector_performance[sector]['symbols_used'] = 5
                
                # Sort sectors by performance
                sorted_sectors = sorted(
                    sector_performance.items(),
                    key=lambda x: x[1]['performance'],
                    reverse=True
                )
                
                performance_by_period[period] = {
                    'start_date': period_start,
                    'end_date': end_date,
                    'sectors': dict(sorted_sectors),
                    'is_generated': True
                }
            
            self.sector_data = performance_by_period
            
            self.logger.info("Generated default sector data based on market index")
            
            return performance_by_period
            
        except Exception as e:
            self.logger.warning(f"Error generating default sector data: {e}")
            return self._generate_random_sector_data(start_date, end_date)
    
    def _generate_random_sector_data(self, start_date, end_date):
        """
        Generate random sector data.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Random sector data
        """
        # Generate random performance data for default sectors
        np.random.seed(42)  # For reproducibility
        
        performance_by_period = {}
        
        for period in self.config['analysis_periods']:
            period_start = end_date - timedelta(days=period)
            
            # Generate random sector performances
            sector_performance = {}
            
            for sector in self.config['default_sectors']:
                performance = (np.random.random() - 0.3) * 0.2  # -6% to +14%
                
                sector_performance[sector] = {
                    'performance': performance,
                    'symbols_count': 5,
                    'symbols_used': 5
                }
            
            # Sort sectors by performance
            sorted_sectors = sorted(
                sector_performance.items(),
                key=lambda x: x[1]['performance'],
                reverse=True
            )
            
            performance_by_period[period] = {
                'start_date': period_start,
                'end_date': end_date,
                'sectors': dict(sorted_sectors),
                'is_generated': True,
                'is_random': True
            }
        
        self.sector_data = performance_by_period
        
        self.logger.info("Generated random sector data")
        
        return performance_by_period
    
    def analyze_sector_rotation(self, start_date=None, end_date=None):
        """
        Analyze sector rotation patterns.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Sector rotation analysis
        """
        self.logger.info("Analyzing sector rotation patterns")
        
        # Get sector data if not already available
        if self.sector_data is None:
            self.get_sector_data(start_date, end_date)
            
        if not self.sector_data:
            self.logger.error("No sector data available for analysis")
            return None
        
        # Prepare analysis results
        analysis = {
            'timestamp': datetime.now(),
            'periods': {},
            'current_stage': None,
            'next_sectors': [],
            'visualizations': {}
        }
        
        # Analyze each period
        for period, data in self.sector_data.items():
            sectors = data['sectors']
            
            # Calculate key metrics
            sector_names = list(sectors.keys())
            performances = [s['performance'] for s in sectors.values()]
            
            # Relative strength
            market_avg = np.mean(performances)
            relative_strength = {
                s: (p['performance'] - market_avg) / abs(market_avg) if market_avg != 0 else 0
                for s, p in sectors.items()
            }
            
            # Trend analysis
            is_bullish = market_avg > 0
            count_positive = sum(1 for p in performances if p > 0)
            count_negative = len(performances) - count_positive
            market_breadth = count_positive / len(performances) if performances else 0
            
            # Sector leadership
            leading_sectors = sector_names[:3]  # Top 3
            lagging_sectors = sector_names[-3:]  # Bottom 3
            
            # Determine market cycle stage based on typical sector rotation
            cycle_stage = self._determine_cycle_stage(sectors)
            
            # Identify next likely leading sectors based on rotation model
            next_sectors = self._predict_next_sectors(cycle_stage, sectors)
            
            # Store period analysis
            analysis['periods'][period] = {
                'start_date': data['start_date'],
                'end_date': data['end_date'],
                'market_avg': market_avg,
                'is_bullish': is_bullish,
                'market_breadth': market_breadth,
                'positive_sectors': count_positive,
                'negative_sectors': count_negative,
                'leading_sectors': leading_sectors,
                'lagging_sectors': lagging_sectors,
                'relative_strength': relative_strength,
                'cycle_stage': cycle_stage
            }
        
        # Determine current market stage (use shortest period)
        shortest_period = min(self.config['analysis_periods'])
        if shortest_period in analysis['periods']:
            analysis['current_stage'] = analysis['periods'][shortest_period]['cycle_stage']
            analysis['next_sectors'] = self._predict_next_sectors(
                analysis['current_stage'],
                self.sector_data[shortest_period]['sectors']
            )
        
        # Generate visualizations
        try:
            visualization_data = self._visualize_sector_rotation()
            
            if visualization_data:
                analysis['visualizations'] = visualization_data
                
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
        
        self.current_stage = analysis['current_stage']
        
        self.logger.info(f"Sector rotation analysis complete: stage = {analysis['current_stage']}")
        
        # Save analysis
        self._save_analysis(analysis)
        
        return analysis
    
    def _determine_cycle_stage(self, sectors):
        """
        Determine the current market cycle stage based on sector performance.
        
        Args:
            sectors (dict): Sector performance data
            
        Returns:
            str: Market cycle stage
        """
        # Get performance for key sectors
        sector_list = list(sectors.keys())
        sector_perf = {s: p['performance'] for s, p in sectors.items()}
        
        # Define key sector groups for reference
        early_expansion = ['Technology', 'Consumer', 'Industrial']
        late_expansion = ['Materials', 'Energy', 'Financial']
        early_contraction = ['Healthcare', 'Utilities', 'Consumer']
        late_contraction = ['Financial', 'Technology', 'Real Estate']
        
        # Check which sectors are leading
        leaders = sector_list[:3]  # Top 3 performing sectors
        
        # Count matches with each stage
        matches = {
            'Early Expansion': sum(1 for s in leaders if any(s.startswith(e) for e in early_expansion)),
            'Late Expansion': sum(1 for s in leaders if any(s.startswith(e) for e in late_expansion)),
            'Early Contraction': sum(1 for s in leaders if any(s.startswith(e) for e in early_contraction)),
            'Late Contraction': sum(1 for s in leaders if any(s.startswith(e) for e in late_contraction))
        }
        
        # Find stage with most matches
        stage = max(matches.items(), key=lambda x: x[1])[0]
        
        # Check overall market trend
        avg_perf = np.mean(list(sector_perf.values()))
        
        # Adjust stage based on overall trend
        if stage in ['Early Expansion', 'Late Expansion'] and avg_perf < 0:
            # If expansion stage but negative returns, likely Early Contraction
            stage = 'Early Contraction'
        elif stage in ['Early Contraction', 'Late Contraction'] and avg_perf > 0:
            # If contraction stage but positive returns, likely Late Expansion
            stage = 'Late Expansion'
        
        return stage
    
    def _predict_next_sectors(self, current_stage, sectors):
        """
        Predict next leading sectors based on rotation model.
        
        Args:
            current_stage (str): Current market cycle stage
            sectors (dict): Current sector performance
            
        Returns:
            list: Next leading sectors
        """
        # Define typical sector rotation model
        rotation_model = {
            'Early Expansion': ['Technology', 'Consumer', 'Industrial'],
            'Late Expansion': ['Materials', 'Energy', 'Financial'],
            'Early Contraction': ['Healthcare', 'Utilities', 'Consumer'],
            'Late Contraction': ['Financial', 'Technology', 'Real Estate']
        }
        
        # Define typical transition flow
        next_stage_map = {
            'Early Expansion': 'Late Expansion',
            'Late Expansion': 'Early Contraction',
            'Early Contraction': 'Late Contraction',
            'Late Contraction': 'Early Expansion'
        }
        
        # Get next stage
        next_stage = next_stage_map.get(current_stage, 'Early Expansion')
        
        # Get sectors for next stage
        next_sectors_template = rotation_model.get(next_stage, [])
        
        # Match with actual sectors in portfolio
        next_sectors = []
        
        for template_sector in next_sectors_template:
            # Find matching sectors
            matches = [s for s in sectors.keys() if template_sector in s]
            
            # Sort by current performance
            matches.sort(key=lambda s: sectors[s]['performance'], reverse=True)
            
            if matches:
                next_sectors.append(matches[0])
        
        return next_sectors
    
    def _visualize_sector_rotation(self):
        """
        Generate visualizations for sector rotation analysis.
        
        Returns:
            dict: Base64 encoded visualization images
        """
        if not self.sector_data:
            return None
            
        visualizations = {}
        
        try:
            # Sector performance bar chart
            shortest_period = min(self.config['analysis_periods'])
            if shortest_period in self.sector_data:
                sectors = self.sector_data[shortest_period]['sectors']
                sector_names = list(sectors.keys())
                performances = [s['performance'] * 100 for s in sectors.values()]  # Convert to percentage
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(sector_names, performances)
                
                # Color bars based on performance
                for i, performance in enumerate(performances):
                    color = 'green' if performance > 0 else 'red'
                    bars[i].set_color(color)
                
                plt.title(f'Sector Performance (Last {shortest_period} Days)')
                plt.ylabel('Performance (%)')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save plot to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                
                # Encode as base64
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                
                visualizations['sector_performance'] = img_str
            
            # Sector performance heatmap for multiple periods
            if len(self.sector_data) > 1:
                # Collect data for all periods
                periods = sorted(self.sector_data.keys())
                all_sectors = set()
                
                for period in periods:
                    all_sectors.update(self.sector_data[period]['sectors'].keys())
                
                all_sectors = sorted(all_sectors)
                
                # Create data matrix
                data = np.zeros((len(all_sectors), len(periods)))
                
                for i, sector in enumerate(all_sectors):
                    for j, period in enumerate(periods):
                        if sector in self.sector_data[period]['sectors']:
                            data[i, j] = self.sector_data[period]['sectors'][sector]['performance'] * 100
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                plt.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)
                
                plt.yticks(range(len(all_sectors)), all_sectors)
                plt.xticks(range(len(periods)), [f"{p} days" for p in periods])
                
                plt.colorbar(label='Performance (%)')
                plt.title('Sector Performance Across Time Periods')
                plt.tight_layout()
                
                # Save plot to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                
                # Encode as base64
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                
                visualizations['sector_heatmap'] = img_str
            
            # Relative strength chart
            shortest_period = min(self.config['analysis_periods'])
            if shortest_period in self.sector_data:
                sectors = self.sector_data[shortest_period]['sectors']
                sector_names = list(sectors.keys())
                
                # Calculate market average
                performances = [s['performance'] for s in sectors.values()]
                market_avg = np.mean(performances)
                
                # Calculate relative strength
                rel_strength = [(s['performance'] - market_avg) / abs(market_avg) * 100 if market_avg != 0 else 0
                               for s in sectors.values()]
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(sector_names, rel_strength)
                
                # Color bars based on relative strength
                for i, strength in enumerate(rel_strength):
                    color = 'green' if strength > 0 else 'red'
                    bars[i].set_color(color)
                
                plt.title(f'Sector Relative Strength (Last {shortest_period} Days)')
                plt.ylabel('Relative Strength (%)')
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save plot to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                
                # Encode as base64
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                
                visualizations['relative_strength'] = img_str
            
            # Sector rotation clock visualization
            plt.figure(figsize=(8, 8))
            
            # Create circle
            circle = plt.Circle((0, 0), 1, fill=False, color='black')
            plt.gca().add_patch(circle)
            
            # Define stages and their angles
            stages = self.config['rotation_stages']
            stage_angles = np.linspace(0, 2*np.pi, len(stages), endpoint=False)
            
            # Plot stage labels
            for stage, angle in zip(stages, stage_angles):
                x = 1.2 * np.cos(angle)
                y = 1.2 * np.sin(angle)
                plt.text(x, y, stage, ha='center', va='center', fontsize=12)
            
            # Highlight current stage
            if self.current_stage:
                current_idx = stages.index(self.current_stage)
                current_angle = stage_angles[current_idx]
                
                # Plot arrow
                x = np.cos(current_angle)
                y = np.sin(current_angle)
                plt.arrow(0, 0, x * 0.9, y * 0.9, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                
                # Add typical sectors for this stage
                typical_sectors = self._get_typical_sectors(self.current_stage)
                sectors_text = ", ".join(typical_sectors)
                
                plt.text(0, -1.4, f"Current Stage: {self.current_stage}", ha='center', fontsize=12, fontweight='bold')
                plt.text(0, -1.55, f"Typical Leaders: {sectors_text}", ha='center', fontsize=10)
            
            # Set equal aspect ratio
            plt.axis('equal')
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.7, 1.5)
            plt.axis('off')
            plt.title('Sector Rotation Clock', fontsize=14)
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Encode as base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            visualizations['rotation_clock'] = img_str
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            return None
    
    def _get_typical_sectors(self, stage):
        """
        Get typical leading sectors for a given stage.
        
        Args:
            stage (str): Market cycle stage
            
        Returns:
            list: Typical leading sectors
        """
        typical_sectors = {
            'Early Expansion': ['Technology', 'Consumer Discretionary', 'Industrials'],
            'Late Expansion': ['Materials', 'Energy', 'Financials'],
            'Early Contraction': ['Healthcare', 'Utilities', 'Consumer Staples'],
            'Late Contraction': ['Financials', 'Technology', 'Real Estate']
        }
        
        return typical_sectors.get(stage, [])
    
    def _save_analysis(self, analysis):
        """
        Save sector rotation analysis to database.
        
        Args:
            analysis (dict): Analysis data
            
        Returns:
            str: Analysis ID
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in analysis:
                analysis['timestamp'] = datetime.now()
            
            # Insert into database
            result = self.db.sector_analysis_collection.insert_one(analysis)
            analysis_id = str(result.inserted_id)
            
            self.logger.info(f"Saved sector rotation analysis with ID: {analysis_id}")
            
            return analysis_id
            
        except Exception as e:
            self.logger.error(f"Error saving analysis: {e}")
            return None
    
    def analyze_symbol(self, symbol, exchange, start_date=None, end_date=None):
        """
        Analyze a symbol in the context of sector rotation.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Symbol analysis
        """
        self.logger.info(f"Analyzing {symbol} {exchange} in sector rotation context")
        
        # Run sector rotation analysis if not already done
        if self.current_stage is None:
            self.analyze_sector_rotation(start_date, end_date)
            
        if not self.sector_data:
            self.logger.error("No sector data available for analysis")
            return None
        
        try:
            # Get symbol data
            symbol_info = self.db.portfolio_collection.find_one({
                'symbol': symbol,
                'exchange': exchange
            })
            
            if not symbol_info:
                self.logger.warning(f"No portfolio data for {symbol} {exchange}")
                return None
                
            # Get sector for this symbol
            sector = symbol_info.get('sector')
            
            if not sector:
                self.logger.warning(f"No sector information for {symbol} {exchange}")
                return None
                
            # Set date range
            end_date = end_date or datetime.now()
            
            # Calculate start dates for different periods
            period_start_dates = {}
            
            for period in self.config['analysis_periods']:
                period_start_dates[period] = end_date - timedelta(days=period)
            
            # Get symbol performance for each period
            symbol_performance = {}
            
            for period, start_date in period_start_dates.items():
                performance = self._calculate_symbol_performance(
                    symbol, exchange, start_date, end_date
                )
                
                if performance is not None:
                    symbol_performance[period] = performance
            
            # Compare to sector performance for each period
            comparison = {}
            
            for period, perf in symbol_performance.items():
                if period in self.sector_data:
                    sector_data = self.sector_data[period]['sectors']
                    
                    if sector in sector_data:
                        sector_perf = sector_data[sector]['performance']
                        
                        # Calculate relative performance
                        relative_perf = perf - sector_perf
                        
                        comparison[period] = {
                            'symbol_performance': perf,
                            'sector_performance': sector_perf,
                            'relative_performance': relative_perf,
                            'outperforming_sector': relative_perf > 0
                        }
            
            # Determine if symbol is in favorable sector
            is_favorable = False
            
            if self.current_stage:
                favorable_sectors = self._get_typical_sectors(self.current_stage)
                is_favorable = any(favorable in sector for favorable in favorable_sectors)
            
            # Calculate correlation with sector
            correlation = self._calculate_sector_correlation(symbol, exchange, sector, end_date)
            
            # Generate analysis
            analysis = {
                'symbol': symbol,
                'exchange': exchange,
                'sector': sector,
                'timestamp': datetime.now(),
                'current_stage': self.current_stage,
                'is_favorable_sector': is_favorable,
                'sector_correlation': correlation,
                'performance_comparison': comparison
            }
            
            # Add recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
            # Save analysis
            self._save_symbol_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing symbol: {e}")
            return None
    
    def _calculate_sector_correlation(self, symbol, exchange, sector, end_date=None):
        """
        Calculate correlation between symbol and its sector.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            sector (str): Sector name
            end_date (datetime): End date
            
        Returns:
            float: Correlation coefficient
        """
        try:
            end_date = end_date or datetime.now()
            start_date = end_date - timedelta(days=90)  # Use 90 days for correlation
            
            # Get symbol data
            symbol_query = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': 'day',
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            symbol_cursor = self.db.market_data_collection.find(symbol_query).sort('timestamp', 1)
            symbol_data = list(symbol_cursor)
            
            if not symbol_data:
                return None
                
            # Get dates and prices
            symbol_dates = [d['timestamp'] for d in symbol_data]
            symbol_prices = [d['close'] for d in symbol_data]
            
            # Get all symbols in the same sector
            sector_query = {
                'sector': sector,
                'status': 'active'
            }
            
            sector_cursor = self.db.portfolio_collection.find(sector_query)
            sector_symbols = [(d['symbol'], d['exchange']) for d in sector_cursor if d['symbol'] != symbol]
            
            if not sector_symbols:
                return None
                
            # Calculate returns for symbol
            symbol_returns = np.diff(symbol_prices) / symbol_prices[:-1]
            
            # Calculate average sector returns (excluding this symbol)
            all_sector_returns = []
            
            for s, e in sector_symbols:
                try:
                    s_query = {
                        'symbol': s,
                        'exchange': e,
                        'timeframe': 'day',
                        'timestamp': {
                            '$gte': start_date,
                            '$lte': end_date
                        }
                    }
                    
                    s_cursor = self.db.market_data_collection.find(s_query).sort('timestamp', 1)
                    s_data = list(s_cursor)
                    
                    if len(s_data) > 1:
                        s_prices = [d['close'] for d in s_data]
                        s_returns = np.diff(s_prices) / s_prices[:-1]
                        
                        all_sector_returns.append(s_returns)
                        
                except Exception as e:
                    self.logger.warning(f"Error calculating returns for {s}: {e}")
            
            if not all_sector_returns:
                return None
                
            # Ensure all return series have the same length by trimming to the shortest
            min_length = min(len(symbol_returns), min(len(r) for r in all_sector_returns))
            
            symbol_returns = symbol_returns[:min_length]
            all_sector_returns = [r[:min_length] for r in all_sector_returns]
            
            # Calculate average sector returns
            sector_returns = np.mean(all_sector_returns, axis=0)
            
            # Calculate correlation
            correlation, _ = np.corrcoef(symbol_returns, sector_returns)[0, 1]
            
            return correlation
            
        except Exception as e:
            self.logger.warning(f"Error calculating sector correlation: {e}")
            return None
    
    def _generate_recommendations(self, analysis):
        """
        Generate recommendations based on sector rotation analysis.
        
        Args:
            analysis (dict): Symbol analysis data
            
        Returns:
            dict: Recommendations
        """
        recommendations = {
            'rating': 'neutral',
            'position_size': 'normal',
            'time_horizon': 'medium',
            'notes': []
        }
        
        try:
            # Check if sector is favorable in current stage
            if analysis['is_favorable_sector']:
                recommendations['notes'].append(
                    f"Symbol is in a sector ({analysis['sector']}) that typically performs well in the current market stage ({analysis['current_stage']})."
                )
                recommendations['rating'] = 'positive'
            else:
                recommendations['notes'].append(
                    f"Symbol is in a sector ({analysis['sector']}) that does not typically lead in the current market stage ({analysis['current_stage']})."
                )
                recommendations['rating'] = 'negative'
            
            # Check correlation with sector
            correlation = analysis.get('sector_correlation')
            
            if correlation is not None:
                if correlation > 0.7:
                    recommendations['notes'].append(
                        f"Symbol shows strong correlation ({correlation:.2f}) with its sector, indicating it follows sector trends closely."
                    )
                elif correlation < 0.3:
                    recommendations['notes'].append(
                        f"Symbol shows weak correlation ({correlation:.2f}) with its sector, suggesting it may be driven by company-specific factors."
                    )
            
            # Check relative performance
            if 'performance_comparison' in analysis:
                # Check shortest period first
                shortest_period = min(analysis['performance_comparison'].keys())
                comparison = analysis['performance_comparison'].get(shortest_period)
                
                if comparison:
                    if comparison.get('outperforming_sector', False):
                        recommendations['notes'].append(
                            f"Symbol is outperforming its sector by {comparison['relative_performance']:.2%} over the past {shortest_period} days."
                        )
                        
                        if recommendations['rating'] == 'positive':
                            recommendations['position_size'] = 'larger'
                    else:
                        recommendations['notes'].append(
                            f"Symbol is underperforming its sector by {-comparison['relative_performance']:.2%} over the past {shortest_period} days."
                        )
                        
                        if recommendations['rating'] == 'negative':
                            recommendations['position_size'] = 'smaller'
            
            # Add time horizon recommendation based on current stage
            if analysis['current_stage'] == 'Early Expansion':
                recommendations['time_horizon'] = 'longer'
                recommendations['notes'].append(
                    "Early expansion phase typically offers longer-term opportunities as trends develop."
                )
            elif analysis['current_stage'] == 'Late Expansion':
                recommendations['time_horizon'] = 'medium'
                recommendations['notes'].append(
                    "Late expansion phase suggests taking profits as market approaches peak."
                )
            elif analysis['current_stage'] == 'Early Contraction':
                recommendations['time_horizon'] = 'shorter'
                recommendations['notes'].append(
                    "Early contraction phase suggests caution and shorter holding periods."
                )
            elif analysis['current_stage'] == 'Late Contraction':
                recommendations['time_horizon'] = 'medium'
                recommendations['notes'].append(
                    "Late contraction phase may offer opportunities in preparation for next expansion."
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {e}")
            return {
                'rating': 'neutral',
                'position_size': 'normal',
                'time_horizon': 'medium',
                'notes': ["Error generating detailed recommendations."]
            }
    
    def _save_symbol_analysis(self, analysis):
        """
        Save symbol sector analysis to database.
        
        Args:
            analysis (dict): Analysis data
            
        Returns:
            str: Analysis ID
        """
        try:
            # Insert into database
            result = self.db.symbol_sector_analysis_collection.insert_one(analysis)
            analysis_id = str(result.inserted_id)
            
            self.logger.info(f"Saved symbol sector analysis with ID: {analysis_id}")
            
            return analysis_id
            
        except Exception as e:
            self.logger.error(f"Error saving symbol analysis: {e}")
            return None
    
    def batch_analyze_symbols(self, symbols_list=None):
        """
        Analyze multiple symbols in sector rotation context.
        
        Args:
            symbols_list (list): List of (symbol, exchange) tuples
            
        Returns:
            dict: Batch analysis results
        """
        if symbols_list is None:
            # Get active symbols from portfolio
            cursor = self.db.portfolio_collection.find({
                'status': 'active',
                'trading_config.enabled': True
            })
            
            symbols_list = [(doc['symbol'], doc['exchange']) for doc in cursor]
        
        self.logger.info(f"Batch analyzing {len(symbols_list)} symbols in sector rotation context")
        
        # Run sector rotation analysis if not already done
        if self.current_stage is None:
            self.analyze_sector_rotation()
            
        if not self.sector_data:
            self.logger.error("No sector data available for analysis")
            return None
        
        results = {}
        favorable_symbols = []
        unfavorable_symbols = []
        
        for symbol, exchange in symbols_list:
            try:
                analysis = self.analyze_symbol(symbol, exchange)
                
                if analysis:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'success',
                        'analysis': analysis
                    }
                    
                    # Categorize based on recommendations
                    if analysis.get('recommendations', {}).get('rating') == 'positive':
                        favorable_symbols.append(analysis)
                    elif analysis.get('recommendations', {}).get('rating') == 'negative':
                        unfavorable_symbols.append(analysis)
                        
                else:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'error',
                        'message': 'Failed to generate analysis'
                    }
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol} {exchange}: {e}")
                results[f"{symbol}_{exchange}"] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        # Generate summary report
        summary = {
            'timestamp': datetime.now(),
            'current_stage': self.current_stage,
            'total_symbols': len(symbols_list),
            'favorable_symbols': len(favorable_symbols),
            'unfavorable_symbols': len(unfavorable_symbols),
            'top_favorable': [
                {
                    'symbol': s['symbol'],
                    'exchange': s['exchange'],
                    'sector': s['sector'],
                    'rating': s.get('recommendations', {}).get('rating')
                } for s in sorted(
                    favorable_symbols,
                    key=lambda x: float(x.get('performance_comparison', {}).get(
                        min(x.get('performance_comparison', {}).keys()), {}).get('relative_performance', 0)
                    ),
                    reverse=True
                )[:5]  # Top 5
            ],
            'top_unfavorable': [
                {
                    'symbol': s['symbol'],
                    'exchange': s['exchange'],
                    'sector': s['sector'],
                    'rating': s.get('recommendations', {}).get('rating')
                } for s in sorted(
                    unfavorable_symbols,
                    key=lambda x: float(x.get('performance_comparison', {}).get(
                        min(x.get('performance_comparison', {}).keys()), {}).get('relative_performance', 0)
                    )
                )[:5]  # Bottom 5
            ]
        }
        
        # Save summary
        try:
            result = self.db.sector_batch_analysis_collection.insert_one(summary)
            summary_id = str(result.inserted_id)
            
            self.logger.info(f"Saved batch analysis summary with ID: {summary_id}")
            summary['_id'] = summary_id
            
        except Exception as e:
            self.logger.error(f"Error saving batch analysis summary: {e}")
        
        return {
            'summary': summary,
            'results': results
        }