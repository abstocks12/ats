# ml/models/reinforcement.py (Simplified Version)
import numpy as np
import pandas as pd
import logging
import pickle
import base64
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class SimpleMarketEnv:
    """Simplified market environment without gym dependency."""
    
    def __init__(self, data, commission=0.0003, initial_balance=10000, window_size=10):
        self.data = data
        self.commission = commission
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance
        self.done = False
        return self._get_observation()
    
    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        prev_portfolio_value = self.portfolio_value
        
        # Execute action (0=Sell, 1=Hold, 2=Buy)
        if action == 0 and self.shares > 0:  # Sell
            sell_value = self.shares * current_price * (1 - self.commission)
            self.balance += sell_value
            self.shares = 0
        elif action == 2 and self.balance > 0:  # Buy
            max_shares = self.balance / (current_price * (1 + self.commission))
            self.shares += int(max_shares)
            self.balance -= int(max_shares) * current_price * (1 + self.commission)
        
        # Update portfolio value
        self.portfolio_value = self.balance + self.shares * current_price
        
        # Calculate reward
        reward = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        return self._get_observation(), reward, self.done, {}
    
    def _get_observation(self):
        # Get window of market data
        market_data = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Simple features: returns, moving averages, RSI
        features = []
        
        # Price returns
        returns = market_data['close'].pct_change().fillna(0)
        features.extend(returns.values)
        
        # Moving averages
        sma_5 = market_data['close'].rolling(5).mean().fillna(market_data['close'])
        features.extend((market_data['close'] / sma_5).fillna(1).values)
        
        # RSI
        rsi = self._calculate_rsi(market_data['close'])
        features.extend(rsi.fillna(50).values)
        
        # Account info
        features.extend([
            self.balance / self.initial_balance,
            self.shares,
            self.portfolio_value / self.initial_balance
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_rsi(self, prices, window=5):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class SimpleQLearningAgent:
    """Simple Q-Learning agent without deep learning."""
    
    def __init__(self, state_size, action_size, learning_rate=0.1, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        
        # Q-table (simplified using state discretization)
        self.q_table = {}
        self.memory = []
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete state for Q-table."""
        # Simple discretization: convert to bins
        discretized = []
        for value in state:
            if value < -0.1:
                discretized.append(0)
            elif value < -0.05:
                discretized.append(1)
            elif value < 0:
                discretized.append(2)
            elif value < 0.05:
                discretized.append(3)
            elif value < 0.1:
                discretized.append(4)
            else:
                discretized.append(5)
        
        return tuple(discretized[:10])  # Use first 10 features only
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_key = self._discretize_state(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0
        
        # Sample from memory
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        total_loss = 0
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            
            state_key = self._discretize_state(state)
            next_state_key = self._discretize_state(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            target = reward
            if not done:
                target += self.gamma * np.max(self.q_table[next_state_key])
            
            current_q = self.q_table[state_key][action]
            self.q_table[state_key][action] += self.learning_rate * (target - current_q)
            
            total_loss += abs(target - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / batch_size


class ReinforcementLearning:
    """Simplified reinforcement learning for market prediction."""
    
    def __init__(self, db_connector, logger=None):
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        self.agent = None
        self.env = None
        self.scaler = StandardScaler()
        self.trained = False
        self.model_params = None
    
    def prepare_data(self, data, window_size=10, test_size=0.2):
        """Prepare data for reinforcement learning."""
        # Scale the data
        data_scaled = pd.DataFrame(
            self.scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        
        # Split into train and test
        split_idx = int(len(data) * (1 - test_size))
        train_data = data_scaled.iloc[:split_idx]
        test_data = data_scaled.iloc[split_idx:]
        
        return train_data, test_data
    
    def build_agent(self, data, window_size=10, learning_rate=0.1, epsilon=1.0,
                   epsilon_min=0.01, epsilon_decay=0.995, gamma=0.95,
                   commission=0.0003, initial_balance=10000):
        """Build simple Q-learning agent and environment."""
        self.logger.info("Building simple Q-learning agent")
        
        # Create environment
        self.env = SimpleMarketEnv(
            data=data,
            commission=commission,
            initial_balance=initial_balance,
            window_size=window_size
        )
        
        # Create agent
        state_size = len(self.env._get_observation())
        action_size = 3  # Sell, Hold, Buy
        
        self.agent = SimpleQLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            gamma=gamma
        )
        
        self.model_params = {
            'window_size': window_size,
            'learning_rate': learning_rate,
            'epsilon': epsilon,
            'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay,
            'gamma': gamma,
            'commission': commission,
            'initial_balance': initial_balance
        }
        
        self.logger.info(f"Agent built with state size {state_size} and action size {action_size}")
        
        return self.agent, self.env
    
    def train(self, episodes=50, max_steps=None, target_update_freq=10, render=False):
        """Train the Q-learning agent."""
        if self.agent is None or self.env is None:
            self.logger.error("Agent not built. Call build_agent() first.")
            return None
        
        self.logger.info(f"Training agent for {episodes} episodes")
        
        results = {
            'episode_rewards': [],
            'portfolio_values': [],
            'losses': []
        }
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            losses = []
            
            step = 0
            while True:
                # Choose action
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Remember experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(self.agent.memory) > 32:
                    loss = self.agent.replay()
                    if loss > 0:
                        losses.append(loss)
                
                state = next_state
                episode_reward += reward
                step += 1
                
                if done or (max_steps and step >= max_steps):
                    break
            
            # Store results
            results['episode_rewards'].append(episode_reward)
            results['portfolio_values'].append(self.env.portfolio_value)
            results['losses'].append(np.mean(losses) if losses else 0)
            
            if episode % 10 == 0:
                self.logger.info(f"Episode {episode+1}/{episodes}: reward={episode_reward:.4f}, portfolio={self.env.portfolio_value:.2f}")
        
        self.trained = True
        return results
    
    def evaluate(self, test_data, episodes=5):
        """Evaluate the trained agent on test data."""
        if not self.trained or self.agent is None:
            self.logger.error("Agent not trained. Train the agent first.")
            return None
        
        self.logger.info(f"Evaluating agent for {episodes} episodes")
        
        # Create test environment
        test_env = SimpleMarketEnv(
            data=test_data,
            commission=self.model_params['commission'],
            initial_balance=self.model_params['initial_balance'],
            window_size=self.model_params['window_size']
        )
        
        results = {
            'episode_rewards': [],
            'portfolio_values': [],
            'actions': []
        }
        
        for episode in range(episodes):
            state = test_env.reset()
            episode_reward = 0
            episode_actions = []
            
            while True:
                # Choose action (no exploration)
                action = self.agent.act(state, training=False)
                
                # Take action
                next_state, reward, done, _ = test_env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_actions.append(action)
                
                if done:
                    break
            
            # Store results
            results['episode_rewards'].append(episode_reward)
            results['portfolio_values'].append(test_env.portfolio_value)
            results['actions'].append(episode_actions)
        
        # Calculate statistics
        final_portfolio_values = np.array(results['portfolio_values'])
        initial_balance = self.model_params['initial_balance']
        
        results['mean_portfolio_value'] = np.mean(final_portfolio_values)
        results['std_portfolio_value'] = np.std(final_portfolio_values)
        results['mean_return'] = np.mean((final_portfolio_values - initial_balance) / initial_balance)
        results['max_return'] = np.max((final_portfolio_values - initial_balance) / initial_balance)
        results['min_return'] = np.min((final_portfolio_values - initial_balance) / initial_balance)
        
        return results
    
    def save_model(self, symbol, exchange, model_name, description=None):
        """Save the trained model to the database."""
        if not self.trained or self.agent is None:
            self.logger.error("No model to save. Train the model first.")
            return None
        
        try:
            # Serialize Q-table and scaler
            model_data = {
                'q_table': self.agent.q_table,
                'epsilon': self.agent.epsilon,
                'scaler': pickle.dumps(self.scaler)
            }
            
            model_base64 = base64.b64encode(pickle.dumps(model_data)).decode('utf-8')
            
            # Create model document
            model_doc = {
                'symbol': symbol,
                'exchange': exchange,
                'model_name': model_name,
                'model_type': 'reinforcement',
                'parameters': self.model_params,
                'model_data': model_base64,
                'created_date': datetime.now(),
                'description': description or f"Simple Q-Learning model for {symbol} {exchange}"
            }
            
            # Insert into database
            result = self.db.models_collection.insert_one(model_doc)
            model_id = str(result.inserted_id)
            
            self.logger.info(f"Model saved to database with ID: {model_id}")
            return model_id
        
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return None
    
    def load_model(self, model_id=None, symbol=None, exchange=None, model_name=None):
        """Load a model from the database."""
        try:
            # Query database
            query = {}
            if model_id:
                from bson.objectid import ObjectId
                query['_id'] = ObjectId(model_id)
            else:
                if symbol:
                    query['symbol'] = symbol
                if exchange:
                    query['exchange'] = exchange
                if model_name:
                    query['model_name'] = model_name
                query['model_type'] = 'reinforcement'
            
            # Find model
            model_doc = self.db.models_collection.find_one(query, sort=[('created_date', -1)])
            
            if not model_doc:
                self.logger.error(f"Model not found: {query}")
                return False
            
            # Load model data
            model_base64 = model_doc['model_data']
            model_data = pickle.loads(base64.b64decode(model_base64))
            
            # Load parameters
            self.model_params = model_doc['parameters']
            
            # Recreate agent
            self.agent = SimpleQLearningAgent(
                state_size=100,  # Approximate
                action_size=3,
                learning_rate=self.model_params['learning_rate'],
                epsilon=model_data['epsilon'],
                epsilon_min=self.model_params['epsilon_min'],
                epsilon_decay=self.model_params['epsilon_decay'],
                gamma=self.model_params['gamma']
            )
            
            # Load Q-table
            self.agent.q_table = model_data['q_table']
            self.agent.epsilon = model_data['epsilon']
            
            # Load scaler
            self.scaler = pickle.loads(model_data['scaler'])
            
            self.trained = True
            self.logger.info(f"Model loaded: {model_doc['model_name']}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def generate_trading_signal(self, symbol, exchange, current_data, save_prediction=True):
        """Generate a trading signal using the Q-learning model."""
        if not self.trained or self.agent is None:
            self.logger.error("Model not trained. Train the model first.")
            return None
        
        try:
            # Scale the data
            current_data_scaled = pd.DataFrame(
                self.scaler.transform(current_data),
                columns=current_data.columns,
                index=current_data.index
            )
            
            # Create environment for prediction
            window_size = self.model_params['window_size']
            
            if len(current_data_scaled) < window_size:
                self.logger.error(f"Not enough data for prediction. Need at least {window_size} samples.")
                return None
            
            # Create prediction environment
            pred_env = SimpleMarketEnv(
                data=current_data_scaled,
                commission=self.model_params['commission'],
                initial_balance=self.model_params['initial_balance'],
                window_size=window_size
            )
            
            # Get current state
            state = pred_env.reset()
            
            # Choose action
            action = self.agent.act(state, training=False)
            
            # Map action to signal
            signal_map = {0: "sell", 1: "hold", 2: "buy"}
            signal = signal_map[action]
            
            # Simple confidence based on Q-values
            state_key = self.agent._discretize_state(state)
            if state_key in self.agent.q_table:
                q_values = self.agent.q_table[state_key]
                confidence = float(np.max(q_values) / (np.sum(np.abs(q_values)) + 1e-8))
            else:
                confidence = 0.5
            
            # Create prediction document
            prediction_doc = {
                'symbol': symbol,
                'exchange': exchange,
                'date': datetime.now(),
                'for_date': datetime.now(),
                'prediction_type': 'trading_signal',
                'signal': signal,
                'confidence': min(1.0, max(0.0, confidence)),
                'model_type': 'reinforcement_simple'
            }
            
            # Save prediction to database
            if save_prediction:
                self.db.predictions_collection.insert_one(prediction_doc)
                self.logger.info(f"Trading signal saved for {symbol} {exchange}: {signal}")
            
            return prediction_doc
        
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return None