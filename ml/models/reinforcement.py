# ml/models/reinforcement.py
import numpy as np
import pandas as pd
import logging
import gym
from gym import spaces
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
import base64
import json
import os

class MarketEnv(gym.Env):
    """Custom Environment for stock trading using reinforcement learning."""
    
    def __init__(self, data, commission=0.0003, initial_balance=10000, window_size=10):
        """
        Initialize the environment.
        
        Args:
            data (DataFrame): Historical OHLCV data
            commission (float): Commission rate
            initial_balance (float): Initial account balance
            window_size (int): Size of observation window
        """
        super(MarketEnv, self).__init__()
        
        self.data = data
        self.commission = commission
        self.initial_balance = initial_balance
        self.window_size = window_size
        
        # Action space: 0 (Sell), 1 (Hold), 2 (Buy)
        self.action_space = spaces.Discrete(3)
        
        # Number of features per time step in the observation window
        num_features = data.shape[1]
        
        # Observation space: window_size time steps of market data, plus account variables
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size * num_features + 3,),  # +3 for balance, shares, portfolio value
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            array: Initial observation
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance
        self.done = False
        
        return self._get_observation()
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): Action index (0=Sell, 1=Hold, 2=Buy)
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        if self.done:
            return self._get_observation(), 0, True, {}
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Get previous portfolio value for calculating reward
        prev_portfolio_value = self.portfolio_value
        
        # Execute action
        if action == 0:  # Sell
            if self.shares > 0:
                # Calculate sell value after commission
                sell_value = self.shares * current_price * (1 - self.commission)
                self.balance += sell_value
                self.shares = 0
        
        elif action == 2:  # Buy
            if self.balance > 0:
                # Calculate max shares that can be bought
                max_shares = self.balance / (current_price * (1 + self.commission))
                # Buy all available
                self.shares += int(max_shares)
                self.balance -= int(max_shares) * current_price * (1 + self.commission)
        
        # Update portfolio value
        self.portfolio_value = self.balance + self.shares * current_price
        
        # Calculate reward (percent change in portfolio value)
        reward = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        return self._get_observation(), reward, self.done, {}
    
    def _get_observation(self):
        """
        Get current observation.
        
        Returns:
            array: Observation vector
        """
        # Get window of market data
        market_data = self.data.iloc[self.current_step - self.window_size:self.current_step].values.flatten()
        
        # Account information
        account_data = np.array([
            self.balance,
            self.shares,
            self.portfolio_value
        ])
        
        # Combine market and account data
        observation = np.concatenate([market_data, account_data])
        
        return observation


class DQNAgent:
    """Deep Q-Network agent for trading."""
    
    def __init__(self, state_size, action_size, batch_size=32, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        """
        Initialize the agent.
        
        Args:
            state_size (int): Size of state/observation space
            action_size (int): Size of action space
            batch_size (int): Batch size for training
            gamma (float): Discount factor
            epsilon (float): Exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
            learning_rate (float): Learning rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Memory for experience replay
        self.memory = []
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """
        Build the neural network model.
        
        Returns:
            Model: Keras model
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target model to match the main model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Add experience to memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose an action.
        
        Args:
            state: Current state
            training (bool): Whether to use epsilon-greedy or greedy policy
            
        Returns:
            int: Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=None):
        """
        Train the agent using experience replay.
        
        Args:
            batch_size (int): Batch size for training
            
        Returns:
            float: Loss value
        """
        batch_size = batch_size or self.batch_size
        
        if len(self.memory) < batch_size:
            return 0
        
        # Sample batch from memory
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # Q-Learning update
        targets = self.model.predict(states)
        q_future = self.target_model.predict(next_states).max(axis=1)
        
        targets[np.arange(batch_size), actions] = rewards + self.gamma * q_future * (1 - dones)
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, batch_size=32, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]
    
    def save(self, model_path):
        """
        Save the model.
        
        Args:
            model_path (str): Path to save the model
        """
        self.model.save(model_path)
    
    def load(self, model_path):
        """
        Load the model.
        
        Args:
            model_path (str): Path to the model
        """
        self.model = load_model(model_path)
        self.target_model = load_model(model_path)


class ReinforcementLearning:
    """Reinforcement learning for market prediction and trading."""
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the reinforcement learning model.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        self.agent = None
        self.env = None
        self.scaler = StandardScaler()
        self.trained = False
        self.model_params = None
    
    def prepare_data(self, data, window_size=10, test_size=0.2):
        """
        Prepare data for reinforcement learning.
        
        Args:
            data (DataFrame): OHLCV data
            window_size (int): Size of observation window
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: (train_data, test_data)
        """
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
    
    def build_agent(self, data, window_size=10, batch_size=32, gamma=0.95, epsilon=1.0,
                   epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001,
                   commission=0.0003, initial_balance=10000):
        """
        Build the reinforcement learning agent and environment.
        
        Args:
            data (DataFrame): Prepared training data
            window_size (int): Size of observation window
            batch_size (int): Batch size for training
            gamma (float): Discount factor
            epsilon (float): Exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
            learning_rate (float): Learning rate
            commission (float): Trading commission rate
            initial_balance (float): Initial account balance
            
        Returns:
            tuple: (agent, env)
        """
        self.logger.info("Building reinforcement learning agent")
        
        # Create environment
        self.env = MarketEnv(
            data=data,
            commission=commission,
            initial_balance=initial_balance,
            window_size=window_size
        )
        
        # Create agent
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            batch_size=batch_size,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            learning_rate=learning_rate
        )
        
        self.model_params = {
            'window_size': window_size,
            'batch_size': batch_size,
            'gamma': gamma,
            'epsilon': epsilon,
            'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay,
            'learning_rate': learning_rate,
            'commission': commission,
            'initial_balance': initial_balance
        }
        
        self.logger.info(f"Agent built with state size {state_size} and action size {action_size}")
        
        return self.agent, self.env
    
    def train(self, episodes=100, max_steps=None, target_update_freq=10, render=False):
        """
        Train the reinforcement learning agent.
        
        Args:
            episodes (int): Number of episodes to train
            max_steps (int): Maximum steps per episode (None for no limit)
            target_update_freq (int): Frequency to update target model
            render (bool): Whether to render the environment
            
        Returns:
            dict: Training results
        """
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
                loss = self.agent.replay()
                if loss > 0:
                    losses.append(loss)
                
                # Update target model periodically
                if step % target_update_freq == 0:
                    self.agent.update_target_model()
                
                state = next_state
                episode_reward += reward
                step += 1
                
                if done or (max_steps and step >= max_steps):
                    break
            
            # Store results
            results['episode_rewards'].append(episode_reward)
            results['portfolio_values'].append(self.env.portfolio_value)
            results['losses'].append(np.mean(losses) if losses else 0)
            
            self.logger.info(f"Episode {episode+1}/{episodes}: reward={episode_reward:.4f}, portfolio={self.env.portfolio_value:.2f}")
        
        self.trained = True
        
        return results
    
    def evaluate(self, test_data, episodes=10):
        """
        Evaluate the trained agent on test data.
        
        Args:
            test_data (DataFrame): Test data
            episodes (int): Number of evaluation episodes
            
        Returns:
            dict: Evaluation results
        """
        if not self.trained or self.agent is None:
            self.logger.error("Agent not trained. Train the agent first.")
            return None
            
        self.logger.info(f"Evaluating agent for {episodes} episodes")
        
        # Create test environment
        test_env = MarketEnv(
            data=test_data,
            commission=self.model_params['commission'],
            initial_balance=self.model_params['initial_balance'],
            window_size=self.model_params['window_size']
        )
        
        results = {
            'episode_rewards': [],
            'portfolio_values': [],
            'actions': [],
            'balance_history': [],
            'shares_history': []
        }
        
        for episode in range(episodes):
            state = test_env.reset()
            episode_reward = 0
            episode_actions = []
            balance_history = [test_env.balance]
            shares_history = [test_env.shares]
            
            while True:
                # Choose action (no exploration)
                action = self.agent.act(state, training=False)
                
                # Take action
                next_state, reward, done, _ = test_env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_actions.append(action)
                balance_history.append(test_env.balance)
                shares_history.append(test_env.shares)
                
                if done:
                    break
            
            # Store results
            results['episode_rewards'].append(episode_reward)
            results['portfolio_values'].append(test_env.portfolio_value)
            results['actions'].append(episode_actions)
            results['balance_history'].append(balance_history)
            results['shares_history'].append(shares_history)
            
            self.logger.info(f"Evaluation episode {episode+1}/{episodes}: reward={episode_reward:.4f}, portfolio={test_env.portfolio_value:.2f}")
        
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
        """
        Save the trained model to the database.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            model_name (str): Name of the model
            description (str): Model description
            
        Returns:
            str: Model ID
        """
        if not self.trained or self.agent is None:
            self.logger.error("No model to save. Train the model first.")
            return None
            
        import tempfile
        import os
        
        # Save model to temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, 'model.h5')
        self.agent.save(temp_path)
        
        # Read the model file
        with open(temp_path, 'rb') as f:
            model_data = f.read()
        
        # Convert to base64
        model_base64 = base64.b64encode(model_data).decode('utf-8')
        
        # Clean up temp file
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        # Create model document
        from datetime import datetime
        
        model_doc = {
            'symbol': symbol,
            'exchange': exchange,
            'model_name': model_name,
            'model_type': 'reinforcement',
            'parameters': self.model_params,
            'model_data': model_base64,
            'created_date': datetime.now(),
            'description': description or f"Reinforcement learning model for {symbol} {exchange}"
        }
        
        # Save scaler
        import pickle
        scaler_bytes = pickle.dumps(self.scaler)
        model_doc['scaler_data'] = base64.b64encode(scaler_bytes).decode('utf-8')
        
        # Insert into database
        result = self.db.models_collection.insert_one(model_doc)
        model_id = str(result.inserted_id)
        
        self.logger.info(f"Model saved to database with ID: {model_id}")
        
        return model_id
    
    def load_model(self, model_id=None, symbol=None, exchange=None, model_name=None):
        """
        Load a model from the database.
        
        Args:
            model_id (str): Model ID
            symbol (str): Trading symbol
            exchange (str): Exchange
            model_name (str): Name of the model
            
        Returns:
            bool: Success/failure
        """
        import pickle
        import tempfile
        import os
        
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
            
        try:
            # Create temp directory and file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, 'model.h5')
            
            # Decode model data
            model_base64 = model_doc['model_data']
            model_data = base64.b64decode(model_base64)
            
            # Write model to file
            with open(temp_path, 'wb') as f:
                f.write(model_data)
            
            # Load model parameters
            self.model_params = model_doc['parameters']
            
            # Create a dummy environment
            # Create a dummy environment
            import pandas as pd
            import numpy as np
            
            # Create a dummy dataframe for initial environment setup
            dummy_data = pd.DataFrame({
                'open': np.random.random(100),
                'high': np.random.random(100),
                'low': np.random.random(100),
                'close': np.random.random(100),
                'volume': np.random.random(100)
            })
            
            # Build environment
            window_size = self.model_params['window_size']
            commission = self.model_params['commission']
            initial_balance = self.model_params['initial_balance']
            
            self.env = MarketEnv(
                data=dummy_data,
                commission=commission,
                initial_balance=initial_balance,
                window_size=window_size
            )
            
            # Create agent
            state_size = self.env.observation_space.shape[0]
            action_size = self.env.action_space.n
            
            self.agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                batch_size=self.model_params['batch_size'],
                gamma=self.model_params['gamma'],
                epsilon=self.model_params['epsilon_min'],  # Use min epsilon for loaded model
                epsilon_min=self.model_params['epsilon_min'],
                epsilon_decay=self.model_params['epsilon_decay'],
                learning_rate=self.model_params['learning_rate']
            )
            
            # Load weights
            self.agent.load(temp_path)
            
            # Clean up temp files
            os.remove(temp_path)
            os.rmdir(temp_dir)
            
            # Load scaler
            scaler_base64 = model_doc['scaler_data']
            scaler_bytes = base64.b64decode(scaler_base64)
            self.scaler = pickle.loads(scaler_bytes)
            
            self.trained = True
            
            self.logger.info(f"Model loaded: {model_doc['model_name']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def generate_trading_signal(self, symbol, exchange, current_data, save_prediction=True):
        """
        Generate a trading signal using the reinforcement learning model.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            current_data (DataFrame): Current market data
            save_prediction (bool): Whether to save prediction to database
            
        Returns:
            dict: Trading signal details
        """
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
            
            # Prepare environment
            pred_env = MarketEnv(
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
            signal_map = {
                0: "sell",
                1: "hold",
                2: "buy"
            }
            
            signal = signal_map[action]
            
            # Calculate confidence based on Q-values
            q_values = self.agent.model.predict(state.reshape(1, -1))[0]
            q_max = np.max(q_values)
            q_sum = np.sum(np.exp(q_values))  # Softmax normalization
            confidence = np.exp(q_max) / q_sum
            
            # Create prediction document
            from datetime import datetime
            
            prediction_doc = {
                'symbol': symbol,
                'exchange': exchange,
                'date': datetime.now(),
                'for_date': datetime.now(),
                'prediction_type': 'trading_signal',
                'signal': signal,
                'confidence': float(confidence),
                'model_type': 'reinforcement',
                'q_values': {
                    'sell': float(q_values[0]),
                    'hold': float(q_values[1]),
                    'buy': float(q_values[2])
                }
            }
            
            # Get current price if available
            current_price = current_data.iloc[-1]['close'] if 'close' in current_data.columns else None
            if current_price:
                prediction_doc['current_price'] = current_price
            
            # Save prediction to database
            if save_prediction:
                self.db.predictions_collection.insert_one(prediction_doc)
                self.logger.info(f"Trading signal saved for {symbol} {exchange}: {signal}")
            
            return prediction_doc
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return None