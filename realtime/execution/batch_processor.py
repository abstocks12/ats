# realtime/execution/batch_processor.py
import logging
import threading
import time
from datetime import datetime, timedelta
import queue

class BatchProcessor:
    """
    Processes orders in batches for efficient execution.
    """
    
    def __init__(self, order_executor, cost_optimizer=None, logger=None):
        """
        Initialize the batch processor.
        
        Args:
            order_executor: Order executor
            cost_optimizer: Cost optimizer
            logger: Logger instance
        """
        self.order_executor = order_executor
        self.cost_optimizer = cost_optimizer
        self.logger = logger or logging.getLogger(__name__)
        
        # State
        self.is_running = False
        self.batch_queue = queue.Queue()
        self.processing_thread = None
        
        # Tracking
        self.active_batches = {}
        self.batch_results = {}
        
        # Configuration
        self.config = {
            'batch_delay': 1.0,  # Delay between batch processing (seconds)
            'batch_size': 5,  # Maximum number of orders in a batch
            'min_batch_interval': 0.5,  # Minimum interval between sending orders in a batch (seconds)
            'batch_timeout': 60,  # Timeout for batch execution (seconds)
            'max_active_batches': 10,  # Maximum number of active batches
            'auto_group_orders': True,  # Automatically group orders for same symbol
            'prioritize_order_types': ['MARKET', 'LIMIT', 'SL', 'SL-M'],  # Order type priority
            'retry_failed_orders': True,  # Retry failed orders
            'max_retries': 3,  # Maximum number of retries for failed orders
            'cancel_batch_on_error': False  # Cancel entire batch if one order fails
        }
        
        self.logger.info("Batch processor initialized")
    
    def set_config(self, config):
        """
        Set batch processor configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated batch processor configuration")
    
    def start(self):
        """
        Start the batch processor.
        
        Returns:
            bool: Success status
        """
        if self.is_running:
            self.logger.warning("Batch processor is already running")
            return False
            
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Batch processor started")
        
        return True
    
    def stop(self, wait=True):
        """
        Stop the batch processor.
        
        Args:
            wait (bool): Wait for processing thread to complete
            
        Returns:
            bool: Success status
        """
        if not self.is_running:
            self.logger.warning("Batch processor is not running")
            return False
            
        self.is_running = False
        
        if wait and self.processing_thread:
            self.processing_thread.join(timeout=30)
            
        self.logger.info("Batch processor stopped")
        
        return True
    
    def add_order_to_batch(self, order, batch_id=None, priority=0):
        """
        Add an order to a batch.
        
        Args:
            order (dict): Order details
            batch_id (str): Batch ID (optional)
            priority (int): Order priority (higher = more important)
            
        Returns:
            str: Batch ID
        """
        # Validate order
        if not order:
            self.logger.error("Invalid order")
            return None
            
        # Generate batch ID if not provided
        if not batch_id:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
        # Add metadata
        order_with_meta = order.copy()
        order_with_meta.update({
            'batch_id': batch_id,
            'priority': priority,
            'added_at': datetime.now(),
            'status': 'QUEUED',
            'attempts': 0
        })
        
        # Add to batch queue
        try:
            self.batch_queue.put(order_with_meta)
            self.logger.info(f"Added order for {order.get('symbol')} to batch {batch_id}")
            
            return batch_id
            
        except Exception as e:
            self.logger.error(f"Error adding order to batch: {e}")
            return None
    
    def add_orders_to_batch(self, orders, batch_id=None):
        """
        Add multiple orders to a batch.
        
        Args:
            orders (list): List of order details
            batch_id (str): Batch ID (optional)
            
        Returns:
            str: Batch ID
        """
        if not orders:
            self.logger.error("No orders provided")
            return None
            
        # Generate batch ID if not provided
        if not batch_id:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
        # Add each order to batch
        successful_adds = 0
        
        for i, order in enumerate(orders):
            # Use reverse index as priority (so first orders have higher priority)
            priority = len(orders) - i
            
            if self.add_order_to_batch(order, batch_id, priority):
                successful_adds += 1
                
        if successful_adds > 0:
            self.logger.info(f"Added {successful_adds}/{len(orders)} orders to batch {batch_id}")
            return batch_id
        else:
            self.logger.error("Failed to add any orders to batch")
            return None
    
    def create_basket_order(self, orders, batch_id=None):
        """
        Create a basket order (multiple orders to be executed together).
        
        Args:
            orders (list): List of order details
            batch_id (str): Batch ID (optional)
            
        Returns:
            str: Batch ID
        """
        if not orders:
            self.logger.error("No orders provided for basket")
            return None
            
        # Generate batch ID if not provided
        if not batch_id:
            batch_id = f"basket_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
        # Group orders by symbol if configured
        if self.config['auto_group_orders']:
            grouped_orders = {}
            
            for order in orders:
                symbol = order.get('symbol')
                exchange = order.get('exchange')
                
                if not symbol or not exchange:
                    continue
                    
                key = f"{symbol}:{exchange}"
                
                if key not in grouped_orders:
                    grouped_orders[key] = []
                    
                grouped_orders[key].append(order)
                
            # Add each group to batch with priority based on order type
            for group_key, group_orders in grouped_orders.items():
                # Sort by order type priority
                sorted_orders = sorted(
                    group_orders,
                    key=lambda o: self.config['prioritize_order_types'].index(o.get('order_type', 'LIMIT')) 
                    if o.get('order_type') in self.config['prioritize_order_types'] else 999
                )
                
                # Add to batch with priority
                for i, order in enumerate(sorted_orders):
                    priority = len(sorted_orders) - i
                    self.add_order_to_batch(order, batch_id, priority)
        else:
            # Add all orders to batch
            self.add_orders_to_batch(orders, batch_id)
            
        # Store batch metadata
        self.active_batches[batch_id] = {
            'total_orders': len(orders),
            'completed_orders': 0,
            'failed_orders': 0,
            'created_at': datetime.now(),
            'status': 'PENDING'
        }
        
        self.logger.info(f"Created basket order {batch_id} with {len(orders)} orders")
        
        return batch_id
    
    def get_batch_status(self, batch_id):
        """
        Get status of a batch.
        
        Args:
            batch_id (str): Batch ID
            
        Returns:
            dict: Batch status
        """
        # Check active batches
        if batch_id in self.active_batches:
            return self.active_batches[batch_id]
            
        # Check completed batches
        if batch_id in self.batch_results:
            return self.batch_results[batch_id]
            
        return None
    
    def cancel_batch(self, batch_id):
        """
        Cancel a batch.
        
        Args:
            batch_id (str): Batch ID
            
        Returns:
            bool: Success status
        """
        # Check if batch exists
        if batch_id not in self.active_batches:
            self.logger.warning(f"Batch {batch_id} not found or already completed")
            return False
            
        # Update batch status
        self.active_batches[batch_id]['status'] = 'CANCELLED'
        
        # Move to results
        self.batch_results[batch_id] = self.active_batches.pop(batch_id)
        
        self.logger.info(f"Cancelled batch {batch_id}")
        
        return True
    
    def _processing_loop(self):
        """
        Main processing loop for batches.
        """
        while self.is_running:
            try:
                # Process batches
                self._process_batch_queue()
                
                # Update batch status
                self._update_batch_status()
                
                # Sleep for batch delay
                time.sleep(self.config['batch_delay'])
                
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}")
                time.sleep(5)  # Sleep longer on error
    
    def _process_batch_queue(self):
        """
        Process orders in the batch queue.
        """
        # Check if we have too many active batches
        if len(self.active_batches) >= self.config['max_active_batches']:
            return
            
        # Collect orders for processing
        orders_to_process = []
        batch_map = {}
        
        # Get up to batch_size orders
        while len(orders_to_process) < self.config['batch_size'] and not self.batch_queue.empty():
            try:
                order = self.batch_queue.get_nowait()
                
                if order:
                    orders_to_process.append(order)
                    
                    # Track which batch this order belongs to
                    batch_id = order.get('batch_id')
                    
                    if batch_id:
                        if batch_id not in batch_map:
                            batch_map[batch_id] = []
                            
                        batch_map[batch_id].append(order)
                        
                self.batch_queue.task_done()
                
            except queue.Empty:
                break
                
        if not orders_to_process:
            return
            
        # Process orders by batch
        for batch_id, batch_orders in batch_map.items():
            # Sort by priority
            batch_orders.sort(key=lambda o: o.get('priority', 0), reverse=True)
            
            # Process batch
            self._process_batch(batch_id, batch_orders)
    
    def _process_batch(self, batch_id, orders):
        """
        Process a batch of orders.
        
        Args:
            batch_id (str): Batch ID
            orders (list): Orders to process
        """
        # Initialize batch status if needed
        if batch_id not in self.active_batches:
            self.active_batches[batch_id] = {
                'total_orders': len(orders),
                'completed_orders': 0,
                'failed_orders': 0,
                'created_at': datetime.now(),
                'status': 'PROCESSING'
            }
            
        # Process each order
        for order in orders:
            try:
                # Update order status
                order['status'] = 'PROCESSING'
                order['processed_at'] = datetime.now()
                
                # Optimize price if cost optimizer is available
                if self.cost_optimizer and order.get('order_type') == 'LIMIT':
                    symbol = order.get('symbol')
                    exchange = order.get('exchange')
                    action = order.get('action')
                    current_price = order.get('price')
                    
                    if symbol and exchange and action and current_price:
                        optimized_price = self.cost_optimizer.optimize_limit_price(
                            symbol=symbol,
                            exchange=exchange,
                            action=action,
                            current_price=current_price,
                            order_type=order.get('order_type'),
                            product_type=order.get('product_type')
                        )
                        
                        if optimized_price:
                            order['price'] = optimized_price
                
                # Execute order
                if self.order_executor:
                    result = self.order_executor.place_order(order)
                    
                    if result:
                        # Order placed successfully
                        order['broker_order_id'] = result.get('broker_order_id')
                        order['status'] = 'PLACED'
                        
                        # Update batch status
                        self.active_batches[batch_id]['completed_orders'] += 1
                    else:
                        # Order failed
                        order['status'] = 'FAILED'
                        order['attempts'] = order.get('attempts', 0) + 1
                        
                        # Retry if configured
                        if (self.config['retry_failed_orders'] and 
                            order.get('attempts', 0) < self.config['max_retries']):
                            # Put back in queue for retry
                            self.batch_queue.put(order)
                        else:
                            # Mark as permanently failed
                            self.active_batches[batch_id]['failed_orders'] += 1
                        
                        # Cancel batch if configured
                        if self.config['cancel_batch_on_error']:
                            self.active_batches[batch_id]['status'] = 'CANCELLED'
                            self.logger.warning(f"Cancelled batch {batch_id} due to order execution error")
                            
                            # Move remaining orders back to queue
                            remaining_orders = [o for o in orders if o.get('status') not in ['PLACED', 'FAILED']]
                            
                            for o in remaining_orders:
                                o['status'] = 'CANCELLED'
                                
                            break
                else:
                    self.logger.error("No order executor available")
                    order['status'] = 'FAILED'
                    self.active_batches[batch_id]['failed_orders'] += 1
                
                # Add small delay between orders
                time.sleep(self.config['min_batch_interval'])
                
            except Exception as e:
                self.logger.error(f"Error processing order in batch {batch_id}: {e}")
                order['status'] = 'FAILED'
                order['error'] = str(e)
                self.active_batches[batch_id]['failed_orders'] += 1
    
    def _update_batch_status(self):
        """
        Update status of active batches.
        """
        # Check each active batch
        for batch_id in list(self.active_batches.keys()):
            batch = self.active_batches[batch_id]
            
            # Check if all orders are processed
            if batch['completed_orders'] + batch['failed_orders'] >= batch['total_orders']:
                # Batch is complete
                if batch['failed_orders'] == 0:
                    batch['status'] = 'COMPLETED'
                elif batch['failed_orders'] == batch['total_orders']:
                    batch['status'] = 'FAILED'
                else:
                    batch['status'] = 'PARTIALLY_COMPLETED'
                    
                batch['completed_at'] = datetime.now()
                
                # Move to results
                self.batch_results[batch_id] = batch
                del self.active_batches[batch_id]
                
                self.logger.info(f"Batch {batch_id} {batch['status']} with {batch['completed_orders']}/{batch['total_orders']} orders")
                
            # Check for timeout
            elif (datetime.now() - batch['created_at']).total_seconds() > self.config['batch_timeout']:
                # Batch has timed out
                batch['status'] = 'TIMEOUT'
                batch['completed_at'] = datetime.now()
                
                # Move to results
                self.batch_results[batch_id] = batch
                del self.active_batches[batch_id]
                
                self.logger.warning(f"Batch {batch_id} timed out with {batch['completed_orders']}/{batch['total_orders']} completed")
    
    def execute_twap_strategy(self, symbol, exchange, action, quantity, price, slices, interval):
        """
        Execute a TWAP (Time-Weighted Average Price) strategy.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            action (str): Action (BUY or SELL)
            quantity (int): Total quantity
            price (float): Base price
            slices (int): Number of slices
            interval (int): Interval between slices (seconds)
            
        Returns:
            str: Batch ID
        """
        if not symbol or not exchange or not action or quantity <= 0 or price <= 0 or slices <= 0:
            self.logger.error("Invalid parameters for TWAP strategy")
            return None
            
        # Calculate slice quantity
        slice_quantity = quantity // slices
        remainder = quantity % slices
        
        # Create orders
        orders = []
        
        for i in range(slices):
            # Add remainder to first slice
            qty = slice_quantity + (remainder if i == 0 else 0)
            
            if qty <= 0:
                continue
                
            # Create order
            order = {
                'symbol': symbol,
                'exchange': exchange,
                'action': action,
                'quantity': qty,
                'price': price,
                'order_type': 'LIMIT',
                'twap_slice': i + 1,
                'execution_delay': i * interval  # Add delay for time-weighted execution
            }
            
            orders.append(order)
            
        # Create batch
        batch_id = f"twap_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add delayed execution
        for order in orders:
            delay = order.pop('execution_delay', 0)
            
            if delay > 0:
                # Schedule with delay
                threading.Timer(delay, lambda o=order: self.add_order_to_batch(o, batch_id)).start()
            else:
                # Add immediately
                self.add_order_to_batch(order, batch_id)
                
        # Initialize batch status
        self.active_batches[batch_id] = {
            'total_orders': len(orders),
            'completed_orders': 0,
            'failed_orders': 0,
            'created_at': datetime.now(),
            'status': 'PENDING',
            'strategy': 'TWAP',
            'symbol': symbol,
            'action': action,
            'total_quantity': quantity
        }
        
        self.logger.info(f"Created TWAP strategy {batch_id} for {symbol} with {slices} slices")
        
        return batch_id
    
    def execute_vwap_strategy(self, symbol, exchange, action, quantity, price, volume_profile):
        """
        Execute a VWAP (Volume-Weighted Average Price) strategy.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            action (str): Action (BUY or SELL)
            quantity (int): Total quantity
            price (float): Base price
            volume_profile (dict): Volume profile (period -> volume percentage)
            
        Returns:
            str: Batch ID
        """
        if (not symbol or not exchange or not action or quantity <= 0 or 
            price <= 0 or not volume_profile):
            self.logger.error("Invalid parameters for VWAP strategy")
            return None
            
        # Create orders based on volume profile
        orders = []
        remaining_quantity = quantity
        
        # Convert volume profile to sorted list
        profile_items = sorted(volume_profile.items())
        
        for period, volume_pct in profile_items:
            # Calculate quantity for this period
            period_quantity = int(quantity * volume_pct)
            
            # Ensure minimum quantity
            period_quantity = max(1, period_quantity)
            
            # Ensure we don't exceed remaining quantity
            period_quantity = min(period_quantity, remaining_quantity)
            
            if period_quantity <= 0:
                continue
                
            # Calculate execution time
            now = datetime.now()
            period_hour = int(period)
            
            # Target time is today at the specified hour
            target_time = datetime(now.year, now.month, now.day, period_hour, 0, 0)
            
            # If target time is in the past, use next day
            if target_time < now:
                target_time = target_time + timedelta(days=1)
                
            # Calculate delay in seconds
            delay_seconds = (target_time - now).total_seconds()
            
            # Create order
            order = {
                'symbol': symbol,
                'exchange': exchange,
                'action': action,
                'quantity': period_quantity,
                'price': price,
                'order_type': 'LIMIT',
                'vwap_period': period,
                'execution_delay': delay_seconds  # Add delay for volume-weighted execution
            }
            
            orders.append(order)
            remaining_quantity -= period_quantity
            
            # Break if we've allocated all quantity
            if remaining_quantity <= 0:
                break
                
        # Add any remaining quantity to the last order
        if remaining_quantity > 0 and orders:
            orders[-1]['quantity'] += remaining_quantity
            
        # Create batch
        batch_id = f"vwap_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add delayed execution
        for order in orders:
            delay = order.pop('execution_delay', 0)
            
            if delay > 0:
                # Schedule with delay
                threading.Timer(delay, lambda o=order: self.add_order_to_batch(o, batch_id)).start()
            else:
                # Add immediately
                self.add_order_to_batch(order, batch_id)
                
        # Initialize batch status
        self.active_batches[batch_id] = {
            'total_orders': len(orders),
            'completed_orders': 0,
            'failed_orders': 0,
            'created_at': datetime.now(),
            'status': 'PENDING',
            'strategy': 'VWAP',
            'symbol': symbol,
            'action': action,
            'total_quantity': quantity
        }
        
        self.logger.info(f"Created VWAP strategy {batch_id} for {symbol} with {len(orders)} slices")
        
        return batch_id
    
    def execute_iceberg_order(self, symbol, exchange, action, quantity, price, visible_size, price_variation=0):
        """
        Execute an iceberg order (large order split into smaller visible portions).
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            action (str): Action (BUY or SELL)
            quantity (int): Total quantity
            price (float): Base price
            visible_size (int): Visible portion size
            price_variation (float): Random price variation (percentage)
            
        Returns:
            str: Batch ID
        """
        if (not symbol or not exchange or not action or quantity <= 0 or 
            price <= 0 or visible_size <= 0):
            self.logger.error("Invalid parameters for iceberg order")
            return None
            
        # Calculate number of slices
        slices = quantity // visible_size
        remainder = quantity % visible_size
        
        if remainder > 0:
            slices += 1
            
        # Create orders
        orders = []
        remaining_quantity = quantity
        
        import random
        
        for i in range(slices):
            # Calculate slice quantity
            slice_qty = min(visible_size, remaining_quantity)
            
            if slice_qty <= 0:
                continue
                
            # Apply random price variation if specified
            slice_price = price
            
            if price_variation > 0:
                variation_factor = 1 + random.uniform(-price_variation, price_variation) / 100
                slice_price = price * variation_factor
                
                # Round to tick size
                if self.cost_optimizer:
                    tick_size = self.cost_optimizer._get_tick_size(symbol, exchange)
                    slice_price = round(slice_price / tick_size) * tick_size
            
            # Create order
            order = {
                'symbol': symbol,
                'exchange': exchange,
                'action': action,
                'quantity': slice_qty,
                'price': slice_price,
                'order_type': 'LIMIT',
                'iceberg_slice': i + 1,
                'total_slices': slices
            }
            
            orders.append(order)
            remaining_quantity -= slice_qty
        
        # Create batch but only add first order initially
        batch_id = f"iceberg_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if orders:
            # Add first order to batch
            first_order = orders[0]
            first_order['is_iceberg_parent'] = True
            self.add_order_to_batch(first_order, batch_id)
            
            # Store remaining orders for later
            self.active_batches[batch_id] = {
                'total_orders': len(orders),
                'completed_orders': 0,
                'failed_orders': 0,
                'created_at': datetime.now(),
                'status': 'PENDING',
                'strategy': 'ICEBERG',
                'symbol': symbol,
                'action': action,
                'total_quantity': quantity,
                'remaining_orders': orders[1:],
                'current_slice': 1
            }
            
            # Start monitoring thread for this iceberg order
            threading.Thread(target=self._monitor_iceberg_order, args=(batch_id,), daemon=True).start()
        
            self.logger.info(f"Created iceberg order {batch_id} for {symbol} with {slices} slices")
            
            return batch_id
        else:
            self.logger.error("No orders created for iceberg strategy")
            return None
    
    def _monitor_iceberg_order(self, batch_id):
        """
        Monitor and continue execution of an iceberg order.
        
        Args:
            batch_id (str): Batch ID
        """
        if batch_id not in self.active_batches:
            return
            
        batch = self.active_batches[batch_id]
        
        if batch.get('strategy') != 'ICEBERG' or not batch.get('remaining_orders'):
            return
            
        # Wait for first order to complete
        while batch_id in self.active_batches and batch['current_slice'] <= 1:
            # Check if batch was cancelled
            if batch.get('status') == 'CANCELLED':
                return
                
            # Sleep briefly
            time.sleep(1)
            
            # Refresh batch data
            if batch_id in self.active_batches:
                batch = self.active_batches[batch_id]
            else:
                return
        
        # Process remaining orders one by one
        while batch_id in self.active_batches and batch.get('remaining_orders'):
            # Check if batch was cancelled
            if batch.get('status') == 'CANCELLED':
                return
                
            # Get next order
            next_order = batch['remaining_orders'].pop(0)
            
            # Update current slice
            batch['current_slice'] += 1
            
            # Add to batch
            self.add_order_to_batch(next_order, batch_id)
            
            # Wait for order to complete
            order_completed = False
            start_time = time.time()
            
            while not order_completed and time.time() - start_time < 300:  # 5-minute timeout
                # Check batch status
                if batch_id in self.active_batches:
                    batch = self.active_batches[batch_id]
                    
                    # Check if order completed
                    if batch['completed_orders'] >= batch['current_slice']:
                        order_completed = True
                        break
                        
                    # Check if batch was cancelled
                    if batch.get('status') == 'CANCELLED':
                        return
                else:
                    # Batch no longer active
                    return
                    
                # Sleep briefly
                time.sleep(1)
                
        # Mark batch as completed if all orders processed
        if batch_id in self.active_batches and not batch.get('remaining_orders'):
            batch['status'] = 'COMPLETED'
            batch['completed_at'] = datetime.now()
            
            # Move to results
            self.batch_results[batch_id] = batch
            del self.active_batches[batch_id]
            
            self.logger.info(f"Iceberg order {batch_id} completed with {batch['completed_orders']}/{batch['total_orders']} orders")