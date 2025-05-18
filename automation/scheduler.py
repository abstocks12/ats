# automation/scheduler.py
import logging
import threading
import time
from datetime import datetime, timedelta
import heapq
import json
import os
import signal
import uuid

class Scheduler:
    """
    Task scheduling system for automated processes.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the scheduler.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # State
        self.is_running = False
        self.scheduler_thread = None
        self.tasks = []  # Priority queue for tasks
        self.task_map = {}  # Map task_id to task
        self.task_lock = threading.RLock()  # Lock for task queue operations
        
        # Timer for periodic execution
        self.timer = None
        
        # Event for waking up the scheduler
        self.wakeup_event = threading.Event()
        
        # Configuration
        self.config = {
            'min_interval': 1,  # Minimum time between task checks (seconds)
            'max_concurrent_tasks': 10,  # Maximum number of concurrent tasks
            'task_timeout': 3600,  # Default task timeout (seconds)
            'retry_failed_tasks': True,  # Retry failed tasks
            'max_retries': 3,  # Maximum number of retries for failed tasks
            'retry_delay': 300,  # Delay between retries (seconds)
            'persistent_storage': True,  # Store tasks in database
            'log_task_output': True,  # Log task output
            'shutdown_timeout': 60,  # Maximum time to wait for running tasks on shutdown (seconds)
            'auto_recover': True,  # Automatically recover tasks from database on startup
            'scheduler_interval': 1  # Seconds between scheduler checks
        }
        
        # Running task tracking
        self.running_tasks = {}  # Map task_id to thread
        self.task_results = {}  # Map task_id to result
        
        # Register signal handlers
        self._register_signal_handlers()
        
        self.logger.info("Scheduler initialized")
    
    def set_config(self, config):
        """
        Set scheduler configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info("Updated scheduler configuration")
    
    def _register_signal_handlers(self):
        """
        Register signal handlers for clean shutdown.
        """
        try:
            signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
            signal.signal(signal.SIGINT, self._handle_shutdown_signal)
            
            self.logger.info("Registered signal handlers")
            
        except (ValueError, AttributeError):
            # Signals not available in some environments (e.g., Windows)
            self.logger.warning("Could not register signal handlers")
    
    def _handle_shutdown_signal(self, signum, frame):
        """
        Handle shutdown signal for clean exit.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.info(f"Received shutdown signal {signum}, stopping scheduler")
        self.stop()
    
    def start(self):
        """
        Start the scheduler.
        
        Returns:
            bool: Success status
        """
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return False
            
        # Recover tasks if configured
        if self.config['auto_recover'] and self.db:
            self._recover_tasks()
            
        # Start scheduler thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        self.logger.info("Scheduler started")
        
        return True
    
    def stop(self, wait=True):
        """
        Stop the scheduler.
        
        Args:
            wait (bool): Wait for scheduler thread to complete
            
        Returns:
            bool: Success status
        """
        if not self.is_running:
            self.logger.warning("Scheduler is not running")
            return False
            
        self.is_running = False
        
        # Wake up scheduler to check is_running
        self.wakeup_event.set()
        
        # Wait for scheduler thread
        if wait and self.scheduler_thread:
            self.scheduler_thread.join(timeout=self.config['shutdown_timeout'])
            
        # Cancel timer if running
        if self.timer:
            self.timer.cancel()
            self.timer = None
            
        # Wait for running tasks to complete
        if wait and self.running_tasks:
            self.logger.info(f"Waiting for {len(self.running_tasks)} running tasks to complete")
            
            # Wait for each task with timeout
            timeout = self.config['shutdown_timeout']
            start_time = time.time()
            
            while self.running_tasks and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            # Log any remaining tasks
            if self.running_tasks:
                self.logger.warning(f"{len(self.running_tasks)} tasks still running after shutdown timeout")
        
        self.logger.info("Scheduler stopped")
        
        return True
    
    def _scheduler_loop(self):
        """
        Main scheduler loop.
        """
        while self.is_running:
            try:
                # Execute due tasks
                self._execute_due_tasks()
                
                # Calculate time until next task
                next_run_time = self._get_next_task_time()
                
                if next_run_time is not None:
                    # Calculate seconds until next task
                    now = datetime.now()
                    wait_seconds = max(0.1, (next_run_time - now).total_seconds())
                    
                    # Wait for next task or wakeup event
                    self.wakeup_event.wait(timeout=min(wait_seconds, self.config['scheduler_interval']))
                    self.wakeup_event.clear()
                else:
                    # No scheduled tasks, wait for wakeup event
                    self.wakeup_event.wait(timeout=self.config['scheduler_interval'])
                    self.wakeup_event.clear()
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(self.config['min_interval'])
    
    def _get_next_task_time(self):
        """
        Get the time of the next scheduled task.
        
        Returns:
            datetime: Time of next task or None if no tasks
        """
        with self.task_lock:
            if not self.tasks:
                return None
                
            # Get time of next task
            next_time, _, _, _ = self.tasks[0]
            
            return next_time
    
    def _execute_due_tasks(self):
        """
        Execute tasks that are due.
        """
        now = datetime.now()
        
        # Check if we're at max concurrent tasks
        if len(self.running_tasks) >= self.config['max_concurrent_tasks']:
            return
            
        with self.task_lock:
            # Get due tasks
            due_tasks = []
            
            while self.tasks and self.tasks[0][0] <= now:
                # Get next task
                task_time, priority, task_id, task = heapq.heappop(self.tasks)
                
                # Check if task is still valid
                if task_id in self.task_map:
                    due_tasks.append((task_id, task))
                    
                    # Remove from task map
                    del self.task_map[task_id]
                    
                    # If we're at max concurrent tasks, break
                    if len(due_tasks) + len(self.running_tasks) >= self.config['max_concurrent_tasks']:
                        break
        
        # Execute due tasks
        for task_id, task in due_tasks:
            self._execute_task(task_id, task)
    
    def _execute_task(self, task_id, task):
        """
        Execute a task.
        
        Args:
            task_id (str): Task ID
            task (dict): Task definition
        """
        try:
            # Get task function and arguments
            func = task.get('func')
            args = task.get('args', [])
            kwargs = task.get('kwargs', {})
            
            if not callable(func):
                self.logger.error(f"Task {task_id} function is not callable")
                return
                
            # Log task execution
            self.logger.info(f"Executing task {task_id}: {task.get('name', 'Unnamed')}")
            
            # Update task status
            task['status'] = 'running'
            task['start_time'] = datetime.now()
            
            if self.config['persistent_storage'] and self.db:
                self._update_task_in_db(task_id, {'status': 'running', 'start_time': task['start_time']})
            
            # Start task in a separate thread
            thread = threading.Thread(target=self._task_wrapper, args=(task_id, task, func, args, kwargs))
            thread.daemon = True
            thread.start()
            
            # Track running task
            self.running_tasks[task_id] = thread
            
        except Exception as e:
            self.logger.error(f"Error starting task {task_id}: {e}")
            
            # Update task status
            task['status'] = 'failed'
            task['error'] = str(e)
            
            if self.config['persistent_storage'] and self.db:
                self._update_task_in_db(task_id, {'status': 'failed', 'error': str(e)})
                
            # Schedule retry if configured
            if self.config['retry_failed_tasks'] and task.get('retries', 0) < self.config['max_retries']:
                self._schedule_retry(task_id, task)
    
    def _task_wrapper(self, task_id, task, func, args, kwargs):
        """
        Wrapper for executing tasks.
        
        Args:
            task_id (str): Task ID
            task (dict): Task definition
            func: Task function
            args (list): Function arguments
            kwargs (dict): Function keyword arguments
        """
        result = None
        error = None
        
        try:
            # Execute task with timeout
            timeout = task.get('timeout', self.config['task_timeout'])
            
            # Use timeout if supported
            if hasattr(threading, 'Thread') and hasattr(threading.Thread, 'join'):
                # Create function wrapper thread that can store result
                result_container = []
                
                def wrapped_func():
                    try:
                        result_container.append(func(*args, **kwargs))
                    except Exception as e:
                        result_container.append(e)
                
                # Start function thread
                func_thread = threading.Thread(target=wrapped_func)
                func_thread.daemon = True
                func_thread.start()
                
                # Wait for thread with timeout
                func_thread.join(timeout=timeout)
                
                # Check if thread completed
                if func_thread.is_alive():
                    # Task timed out
                    error = f"Task timed out after {timeout} seconds"
                    self.logger.error(f"Task {task_id} timed out")
                elif result_container:
                    # Get result
                    if isinstance(result_container[0], Exception):
                        # Task raised exception
                        error = str(result_container[0])
                        self.logger.error(f"Task {task_id} failed: {error}")
                    else:
                        # Task completed successfully
                        result = result_container[0]
            else:
                # No timeout support, just run function
                result = func(*args, **kwargs)
                
        except Exception as e:
            # Task failed
            error = str(e)
            self.logger.error(f"Task {task_id} failed: {error}")
            
        finally:
            # Update task status
            task['end_time'] = datetime.now()
            
            if error:
                task['status'] = 'failed'
                task['error'] = error
                
                if self.config['persistent_storage'] and self.db:
                    self._update_task_in_db(task_id, {'status': 'failed', 'error': error, 'end_time': task['end_time']})
                    
                # Schedule retry if configured
                if self.config['retry_failed_tasks'] and task.get('retries', 0) < self.config['max_retries']:
                    self._schedule_retry(task_id, task)
            else:
                task['status'] = 'completed'
                
                if self.config['persistent_storage'] and self.db:
                    self._update_task_in_db(task_id, {'status': 'completed', 'end_time': task['end_time']})
                
                # Store result
                self.task_results[task_id] = result
                
                # Handle recurring task
                if task.get('recurring'):
                    self._reschedule_recurring_task(task_id, task)
            
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def _schedule_retry(self, task_id, task):
        """
        Schedule a retry for a failed task.
        
        Args:
            task_id (str): Task ID
            task (dict): Task definition
        """
        try:
            # Increment retry count
            retries = task.get('retries', 0) + 1
            task['retries'] = retries
            
            # Calculate retry time
            retry_delay = task.get('retry_delay', self.config['retry_delay'])
            retry_time = datetime.now() + timedelta(seconds=retry_delay)
            
            # Create new task
            retry_task = task.copy()
            retry_task['scheduled_time'] = retry_time
            
            # Schedule retry
            self.logger.info(f"Scheduling retry {retries}/{self.config['max_retries']} for task {task_id} at {retry_time}")
            
            # Generate new task ID for retry
            retry_id = f"{task_id}_retry{retries}"
            
            # Schedule retry task
            self.schedule_task(
                func=task['func'],
                args=task.get('args', []),
                kwargs=task.get('kwargs', {}),
                scheduled_time=retry_time,
                task_id=retry_id,
                name=f"{task.get('name', 'Unnamed')} (Retry {retries})",
                retries=retries
            )
            
        except Exception as e:
            self.logger.error(f"Error scheduling retry for task {task_id}: {e}")
    
    def _reschedule_recurring_task(self, task_id, task):
        """
        Reschedule a recurring task.
        
        Args:
            task_id (str): Task ID
            task (dict): Task definition
        """
        try:
            # Get scheduling parameters
            interval = task.get('interval')
            cron = task.get('cron')
            
            if not interval and not cron:
                self.logger.error(f"Cannot reschedule task {task_id}: no interval or cron specified")
                return
                
            # Calculate next run time
            if interval:
                # Calculate from last scheduled time
                last_time = task.get('scheduled_time', datetime.now())
                next_time = last_time + timedelta(seconds=interval)
                
                # If next time is in the past, calculate from now
                if next_time < datetime.now():
                    next_time = datetime.now() + timedelta(seconds=interval)
            else:
                # Use cron expression
                next_time = self._get_next_cron_time(cron)
                
                if next_time is None:
                    self.logger.error(f"Cannot reschedule task {task_id}: invalid cron expression")
                    return
            
            # Create new task
            next_task = task.copy()
            next_task['scheduled_time'] = next_time
            next_task['status'] = 'scheduled'
            
            if 'start_time' in next_task:
                del next_task['start_time']
                
            if 'end_time' in next_task:
                del next_task['end_time']
                
            if 'error' in next_task:
                del next_task['error']
            
            # Schedule next task
            self.logger.info(f"Rescheduling recurring task {task_id} for {next_time}")
            
            # Generate new task ID for next run
            runs = task.get('runs', 1) + 1
            next_id = f"{task_id.split('_run')[0]}_run{runs}"
            
            # Update run count
            next_task['runs'] = runs
            
            # Check if max runs reached
            max_runs = task.get('max_runs')
            
            if max_runs and runs >= max_runs:
                self.logger.info(f"Recurring task {task_id} reached maximum runs ({max_runs})")
                return
            
            # Schedule next task
            self.schedule_task(
                func=task['func'],
                args=task.get('args', []),
                kwargs=task.get('kwargs', {}),
                scheduled_time=next_time,
                task_id=next_id,
                name=task.get('name', 'Unnamed'),
                recurring=True,
                interval=interval,
                cron=cron,
                max_runs=max_runs,
                runs=runs
            )
            
        except Exception as e:
            self.logger.error(f"Error rescheduling recurring task {task_id}: {e}")
    
    def _get_next_cron_time(self, cron_expression):
        """
        Calculate the next run time based on a cron expression.
        
        Args:
            cron_expression (str): Cron expression
            
        Returns:
            datetime: Next run time or None if invalid
        """
        try:
            # Import croniter if available
            try:
                from croniter import croniter
                
                # Calculate next run time
                base = datetime.now()
                cron = croniter(cron_expression, base)
                next_time = cron.get_next(datetime)
                
                return next_time
                
            except ImportError:
                self.logger.error("Croniter library not installed. Install with 'pip install croniter'")
                
                # Simple implementation for basic cron expressions
                return self._simple_cron_next_time(cron_expression)
                
        except Exception as e:
            self.logger.error(f"Error calculating next cron time: {e}")
            return None
    
    def _simple_cron_next_time(self, cron_expression):
        """
        Simple implementation for calculating next run time from cron expression.
        Only supports basic expressions like "0 9 * * *" (daily at 9:00 AM).
        
        Args:
            cron_expression (str): Cron expression
            
        Returns:
            datetime: Next run time or None if invalid
        """
        try:
            # Parse cron expression
            parts = cron_expression.split()
            
            if len(parts) != 5:
                return None
                
            minute, hour, day, month, weekday = parts
            
            # Get current time
            now = datetime.now()
            
            # Calculate next run time (only basic support)
            if minute == '*' or hour == '*':
                # For expressions with wildcards, default to next minute
                next_time = now + timedelta(minutes=1)
            else:
                # For specific time, calculate next occurrence
                try:
                    minute_val = int(minute)
                    hour_val = int(hour)
                    
                    next_time = now.replace(hour=hour_val, minute=minute_val, second=0, microsecond=0)
                    
                    # If time is in the past, move to tomorrow
                    if next_time <= now:
                        next_time += timedelta(days=1)
                        
                except ValueError:
                    return None
            
            return next_time
            
        except Exception as e:
            self.logger.error(f"Error in simple cron calculation: {e}")
            return None
    
    def schedule_task(self, func, args=None, kwargs=None, scheduled_time=None, task_id=None, 
                     name=None, priority=0, timeout=None, recurring=False, interval=None, 
                     cron=None, max_runs=None, runs=1, retries=0):
        """
        Schedule a task.
        
        Args:
            func: Task function
            args (list): Function arguments
            kwargs (dict): Function keyword arguments
            scheduled_time (datetime): Scheduled execution time
            task_id (str): Task ID (optional)
            name (str): Task name (optional)
            priority (int): Task priority (lower value = higher priority)
            timeout (int): Task timeout in seconds
            recurring (bool): Whether the task recurs
            interval (int): Recurrence interval in seconds
            cron (str): Cron expression for recurrence
            max_runs (int): Maximum number of runs for recurring tasks
            runs (int): Current run count
            retries (int): Current retry count
            
        Returns:
            str: Task ID
        """
        try:
            # Validate function
            if not callable(func):
                self.logger.error("Task function must be callable")
                return None
                
            # Default arguments
            args = args or []
            kwargs = kwargs or {}
            
            # Default scheduled time to now
            if scheduled_time is None:
                scheduled_time = datetime.now()
                
            # Generate task ID if not provided
            if task_id is None:
                task_id = str(uuid.uuid4())
                
            # Default name to function name
            if name is None:
                name = func.__name__ if hasattr(func, '__name__') else 'Unnamed Task'
                
            # Default timeout
            if timeout is None:
                timeout = self.config['task_timeout']
                
            # Validate recurring parameters
            if recurring and not (interval or cron):
                self.logger.error("Recurring tasks require interval or cron expression")
                return None
                
            # Create task
            task = {
                'func': func,
                'args': args,
                'kwargs': kwargs,
                'scheduled_time': scheduled_time,
                'task_id': task_id,
                'name': name,
                'priority': priority,
                'timeout': timeout,
                'status': 'scheduled',
                'recurring': recurring,
                'retries': retries,
                'created_at': datetime.now()
            }
            
            # Add recurring parameters if needed
            if recurring:
                task['interval'] = interval
                task['cron'] = cron
                task['max_runs'] = max_runs
                task['runs'] = runs
            
            # Add to task queue
            with self.task_lock:
                # Store in task map
                self.task_map[task_id] = task
                
                # Add to priority queue
                heapq.heappush(self.tasks, (scheduled_time, priority, task_id, task))
            
            # Store in database if configured
            if self.config['persistent_storage'] and self.db:
                self._store_task_in_db(task_id, task)
                
            self.logger.info(f"Scheduled task {task_id} ({name}) for {scheduled_time}")
            
            # Wake up scheduler to check new task
            self.wakeup_event.set()
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error scheduling task: {e}")
            return None
    
    def schedule_at(self, func, scheduled_time, args=None, kwargs=None, task_id=None, 
                   name=None, priority=0, timeout=None):
        """
        Schedule a task to run at a specific time.
        
        Args:
            func: Task function
            scheduled_time (datetime): Scheduled execution time
            args (list): Function arguments
            kwargs (dict): Function keyword arguments
            task_id (str): Task ID (optional)
            name (str): Task name (optional)
            priority (int): Task priority (lower value = higher priority)
            timeout (int): Task timeout in seconds
            
        Returns:
            str: Task ID
        """
        return self.schedule_task(
            func=func,
            args=args,
            kwargs=kwargs,
            scheduled_time=scheduled_time,
            task_id=task_id,
            name=name,
            priority=priority,
            timeout=timeout
        )
    
    def schedule_in(self, func, delay, args=None, kwargs=None, task_id=None, 
                   name=None, priority=0, timeout=None):
        """
        Schedule a task to run after a delay.
        
        Args:
            func: Task function
            delay (int): Delay in seconds
            args (list): Function arguments
            kwargs (dict): Function keyword arguments
            task_id (str): Task ID (optional)
            name (str): Task name (optional)
            priority (int): Task priority (lower value = higher priority)
            timeout (int): Task timeout in seconds
            
        Returns:
            str: Task ID
        """
        scheduled_time = datetime.now() + timedelta(seconds=delay)
        
        return self.schedule_task(
            func=func,
            args=args,
            kwargs=kwargs,
            scheduled_time=scheduled_time,
            task_id=task_id,
            name=name,
            priority=priority,
            timeout=timeout
        )
    
    def schedule_recurring(self, func, interval=None, cron=None, args=None, kwargs=None, 
                          task_id=None, name=None, priority=0, timeout=None, max_runs=None, 
                          start_time=None):
        """
        Schedule a recurring task.
        
        Args:
            func: Task function
            interval (int): Recurrence interval in seconds
            cron (str): Cron expression for recurrence
            args (list): Function arguments
            kwargs (dict): Function keyword arguments
            task_id (str): Task ID (optional)
            name (str): Task name (optional)
            priority (int): Task priority (lower value = higher priority)
            timeout (int): Task timeout in seconds
            max_runs (int): Maximum number of runs
            start_time (datetime): First execution time (optional)
            
        Returns:
            str: Task ID
        """
        # Validate parameters
        if not interval and not cron:
            self.logger.error("Recurring tasks require interval or cron expression")
            return None
            
        # Calculate start time if not provided
        if start_time is None:
            if interval:
                # Start immediately for interval-based tasks
                start_time = datetime.now()
            else:
                # Calculate from cron expression
                start_time = self._get_next_cron_time(cron)
                
                if start_time is None:
                    self.logger.error("Invalid cron expression")
                    return None
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"{str(uuid.uuid4())}_run1"
            
        return self.schedule_task(
            func=func,
            args=args,
            kwargs=kwargs,
            scheduled_time=start_time,
            task_id=task_id,
            name=name,
            priority=priority,
            timeout=timeout,
            recurring=True,
            interval=interval,
            cron=cron,
            max_runs=max_runs,
            runs=1
        )
    
    def schedule_daily(self, func, time_str, args=None, kwargs=None, task_id=None, 
                      name=None, priority=0, timeout=None, max_runs=None):
        """
        Schedule a task to run daily at a specific time.
        
        Args:
            func: Task function
            time_str (str): Time in "HH:MM" format
            args (list): Function arguments
            kwargs (dict): Function keyword arguments
            task_id (str): Task ID (optional)
            name (str): Task name (optional)
            priority (int): Task priority (lower value = higher priority)
            timeout (int): Task timeout in seconds
            max_runs (int): Maximum number of runs
            
        Returns:
            str: Task ID
        """
        try:
            # Parse time string
            hour, minute = map(int, time_str.split(':'))
            
            # Validate time
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                self.logger.error(f"Invalid time: {time_str}")
                return None
                
            # Create cron expression for daily at specified time
            cron = f"{minute} {hour} * * *"
            
            return self.schedule_recurring(
                func=func,
                cron=cron,
                args=args,
                kwargs=kwargs,
                task_id=task_id,
                name=name or f"Daily at {time_str}",
                priority=priority,
                timeout=timeout,
                max_runs=max_runs
            )
            
        except Exception as e:
            self.logger.error(f"Error scheduling daily task: {e}")
            return None
    
    def schedule_weekly(self, func, day_of_week, time_str, args=None, kwargs=None, 
                       task_id=None, name=None, priority=0, timeout=None, max_runs=None):
        """
        Schedule a task to run weekly on a specific day and time.
        
        Args:
            func: Task function
            day_of_week (int): Day of week (0=Monday, 6=Sunday)
            time_str (str): Time in "HH:MM" format
            args (list): Function arguments
            kwargs (dict): Function keyword arguments
            task_id (str): Task ID (optional)
            name (str): Task name (optional)
            priority (int): Task priority (lower value = higher priority)
            timeout (int): Task timeout in seconds
            max_runs (int): Maximum number of runs
            
        Returns:
            str: Task ID
        """
        try:
            # Parse time string
            hour, minute = map(int, time_str.split(':'))
            
            # Validate time
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                self.logger.error(f"Invalid time: {time_str}")
                return None
                
            # Validate day of week
            if day_of_week < 0 or day_of_week > 6:
                self.logger.error(f"Invalid day of week: {day_of_week}")
                return None
                
            # Convert to cron format (0=Sunday in cron)
            cron_dow = (day_of_week + 1) % 7
                
            # Create cron expression for weekly at specified day and time
            cron = f"{minute} {hour} * * {cron_dow}"
            
            # Day names for log
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            return self.schedule_recurring(
                func=func,
                cron=cron,
                args=args,
                kwargs=kwargs,
                task_id=task_id,
                name=name or f"Weekly on {day_names[day_of_week]} at {time_str}",
                priority=priority,
                timeout=timeout,
                max_runs=max_runs
            )
            
        except Exception as e:
            self.logger.error(f"Error scheduling weekly task: {e}")
            return None
    
    def schedule_monthly(self, func, day_of_month, time_str, args=None, kwargs=None, 
                    task_id=None, name=None, priority=0, timeout=None, max_runs=None):
        """
        Schedule a task to run monthly on a specific day and time.
        
        Args:
            func: Task function
            day_of_month (int): Day of month (1-31)
            time_str (str): Time in "HH:MM" format
            args (list): Function arguments
            kwargs (dict): Function keyword arguments
            task_id (str): Task ID (optional)
            name (str): Task name (optional)
            priority (int): Task priority (lower value = higher priority)
            timeout (int): Task timeout in seconds
            max_runs (int): Maximum number of runs
            
        Returns:
            str: Task ID
        """
        try:
            # Parse time string
            hour, minute = map(int, time_str.split(':'))
            
            # Validate time
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                self.logger.error(f"Invalid time: {time_str}")
                return None
                
            # Validate day of month
            if day_of_month < 1 or day_of_month > 31:
                self.logger.error(f"Invalid day of month: {day_of_month}")
                return None
                
            # Create cron expression for monthly at specified day and time
            cron = f"{minute} {hour} {day_of_month} * *"
            
            return self.schedule_recurring(
                func=func,
                cron=cron,
                args=args,
                kwargs=kwargs,
                task_id=task_id,
                name=name or f"Monthly on day {day_of_month} at {time_str}",
                priority=priority,
                timeout=timeout,
                max_runs=max_runs
            )
            
        except Exception as e:
            self.logger.error(f"Error scheduling monthly task: {e}")
            return None

    def cancel_task(self, task_id):
        """
        Cancel a scheduled task.
        
        Args:
            task_id (str): Task ID
            
        Returns:
            bool: Success status
        """
        try:
            # Check if task exists
            if task_id not in self.task_map:
                self.logger.warning(f"Task {task_id} not found")
                return False
                
            with self.task_lock:
                # Remove from task map
                task = self.task_map.pop(task_id)
                
                # We can't directly remove from priority queue without rebuilding it
                # Mark task as cancelled, it will be skipped when executed
                task['status'] = 'cancelled'
                
                # Remove from database if configured
                if self.config['persistent_storage'] and self.db:
                    self._update_task_in_db(task_id, {'status': 'cancelled'})
                    
            self.logger.info(f"Cancelled task {task_id}: {task.get('name', 'Unnamed')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling task {task_id}: {e}")
            return False

    def get_task_status(self, task_id):
        """
        Get status of a task.
        
        Args:
            task_id (str): Task ID
            
        Returns:
            dict: Task status or None if not found
        """
        try:
            # Check running tasks
            if task_id in self.running_tasks:
                return {'status': 'running', 'task_id': task_id}
                
            # Check task map
            if task_id in self.task_map:
                task = self.task_map[task_id]
                return {
                    'status': task.get('status', 'unknown'),
                    'task_id': task_id,
                    'name': task.get('name', 'Unnamed'),
                    'scheduled_time': task.get('scheduled_time'),
                    'priority': task.get('priority', 0)
                }
                
            # Check database if configured
            if self.config['persistent_storage'] and self.db:
                task = self._get_task_from_db(task_id)
                
                if task:
                    return {
                        'status': task.get('status', 'unknown'),
                        'task_id': task_id,
                        'name': task.get('name', 'Unnamed'),
                        'scheduled_time': task.get('scheduled_time'),
                        'priority': task.get('priority', 0),
                        'start_time': task.get('start_time'),
                        'end_time': task.get('end_time'),
                        'error': task.get('error')
                    }
                    
            # Check task results
            if task_id in self.task_results:
                return {'status': 'completed', 'task_id': task_id, 'has_result': True}
                
            # Task not found
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting task status for {task_id}: {e}")
            return None

    def get_task_result(self, task_id):
        """
        Get result of a completed task.
        
        Args:
            task_id (str): Task ID
            
        Returns:
            Result of task or None if not found or not completed
        """
        try:
            # Check if task has a result
            if task_id in self.task_results:
                return self.task_results[task_id]
                
            # Task not found or not completed
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting task result for {task_id}: {e}")
            return None

    def list_tasks(self, status=None):
        """
        List all tasks with optional status filter.
        
        Args:
            status (str): Filter by status (scheduled, running, completed, failed, cancelled)
            
        Returns:
            list: List of task info dictionaries
        """
        try:
            tasks = []
            
            # Add tasks from task map
            with self.task_lock:
                for task_id, task in self.task_map.items():
                    if status is None or task.get('status') == status:
                        tasks.append({
                            'task_id': task_id,
                            'name': task.get('name', 'Unnamed'),
                            'status': task.get('status', 'unknown'),
                            'scheduled_time': task.get('scheduled_time'),
                            'priority': task.get('priority', 0),
                            'recurring': task.get('recurring', False)
                        })
                        
            # Add running tasks
            if status is None or status == 'running':
                for task_id in self.running_tasks:
                    if task_id not in self.task_map:
                        tasks.append({
                            'task_id': task_id,
                            'status': 'running'
                        })
                        
            # If database is configured, get tasks from there too
            if self.config['persistent_storage'] and self.db:
                db_tasks = self._list_tasks_from_db(status)
                
                # Add tasks not already in the list
                task_ids = {t['task_id'] for t in tasks}
                
                for task in db_tasks:
                    if task['task_id'] not in task_ids:
                        tasks.append(task)
                        
            return tasks
            
        except Exception as e:
            self.logger.error(f"Error listing tasks: {e}")
            return []

    def _store_task_in_db(self, task_id, task):
        """
        Store task in database.
        
        Args:
            task_id (str): Task ID
            task (dict): Task definition
        """
        try:
            if not self.db:
                return
                
            # Create storable task
            db_task = {
                'task_id': task_id,
                'name': task.get('name', 'Unnamed'),
                'status': task.get('status', 'scheduled'),
                'scheduled_time': task.get('scheduled_time'),
                'priority': task.get('priority', 0),
                'recurring': task.get('recurring', False),
                'created_at': task.get('created_at', datetime.now())
            }
            
            # Add recurring parameters if needed
            if task.get('recurring'):
                db_task['interval'] = task.get('interval')
                db_task['cron'] = task.get('cron')
                db_task['max_runs'] = task.get('max_runs')
                db_task['runs'] = task.get('runs', 1)
                
            # Store task definition if serializable
            try:
                # We can't store the function directly, so store the name
                func = task.get('func')
                if func:
                    db_task['func_name'] = func.__name__ if hasattr(func, '__name__') else str(func)
                    
                # Store args and kwargs if serializable
                db_task['args'] = json.dumps(task.get('args', []))
                db_task['kwargs'] = json.dumps(task.get('kwargs', {}))
                
            except (TypeError, ValueError):
                # If not serializable, just store basic info
                self.logger.warning(f"Could not serialize task {task_id} function or arguments")
                
            # Store in database
            self.db.tasks_collection.update_one(
                {'task_id': task_id},
                {'$set': db_task},
                upsert=True
            )
            
        except Exception as e:
            self.logger.error(f"Error storing task {task_id} in database: {e}")

    def _update_task_in_db(self, task_id, updates):
        """
        Update task in database.
        
        Args:
            task_id (str): Task ID
            updates (dict): Task updates
        """
        try:
            if not self.db:
                return
                
            # Update task in database
            self.db.tasks_collection.update_one(
                {'task_id': task_id},
                {'$set': updates}
            )
            
        except Exception as e:
            self.logger.error(f"Error updating task {task_id} in database: {e}")

    def _get_task_from_db(self, task_id):
        """
        Get task from database.
        
        Args:
            task_id (str): Task ID
            
        Returns:
            dict: Task definition or None if not found
        """
        try:
            if not self.db:
                return None
                
            # Get task from database
            task = self.db.tasks_collection.find_one({'task_id': task_id})
            
            return task
            
        except Exception as e:
            self.logger.error(f"Error getting task {task_id} from database: {e}")
            return None

    def _list_tasks_from_db(self, status=None):
        """
        List tasks from database.
        
        Args:
            status (str): Filter by status
            
        Returns:
            list: List of task info dictionaries
        """
        try:
            if not self.db:
                return []
                
            # Build query
            query = {}
            
            if status:
                query['status'] = status
                
            # Get tasks from database
            cursor = self.db.tasks_collection.find(query)
            
            # Convert to list
            tasks = list(cursor)
            
            return tasks
            
        except Exception as e:
            self.logger.error(f"Error listing tasks from database: {e}")
            return []

    def _recover_tasks(self):
        """
        Recover tasks from database.
        """
        try:
            if not self.db:
                return
                
            self.logger.info("Recovering tasks from database")
            
            # Get scheduled and running tasks
            tasks = list(self.db.tasks_collection.find({
                'status': {'$in': ['scheduled', 'running']}
            }))
            
            if not tasks:
                self.logger.info("No tasks to recover")
                return
                
            self.logger.info(f"Found {len(tasks)} tasks to recover")
            
            # Function registry for recovery
            function_registry = self._get_function_registry()
            
            # Recover each task
            for task in tasks:
                try:
                    task_id = task.get('task_id')
                    
                    # Skip if already in task map
                    if task_id in self.task_map:
                        continue
                        
                    # Get function
                    func_name = task.get('func_name')
                    func = function_registry.get(func_name)
                    
                    if not func:
                        self.logger.warning(f"Could not recover task {task_id}: function {func_name} not found")
                        continue
                        
                    # Parse args and kwargs
                    try:
                        args = json.loads(task.get('args', '[]'))
                        kwargs = json.loads(task.get('kwargs', '{}'))
                    except (TypeError, ValueError):
                        args = []
                        kwargs = {}
                        
                    # Get scheduled time
                    scheduled_time = task.get('scheduled_time')
                    
                    if not scheduled_time:
                        self.logger.warning(f"Could not recover task {task_id}: missing scheduled time")
                        continue
                        
                    # Reschedule
                    priority = task.get('priority', 0)
                    name = task.get('name', 'Recovered Task')
                    recurring = task.get('recurring', False)
                    
                    # Create new task
                    new_task = {
                        'func': func,
                        'args': args,
                        'kwargs': kwargs,
                        'scheduled_time': scheduled_time,
                        'task_id': task_id,
                        'name': name,
                        'priority': priority,
                        'status': 'scheduled',
                        'created_at': task.get('created_at', datetime.now())
                    }
                    
                    # Add recurring parameters if needed
                    if recurring:
                        new_task['recurring'] = True
                        new_task['interval'] = task.get('interval')
                        new_task['cron'] = task.get('cron')
                        new_task['max_runs'] = task.get('max_runs')
                        new_task['runs'] = task.get('runs', 1)
                    
                    # If task was running, check if it should be rescheduled
                    if task.get('status') == 'running':
                        # If task has been running for longer than timeout, reschedule
                        start_time = task.get('start_time')
                        
                        if start_time:
                            elapsed = (datetime.now() - start_time).total_seconds()
                            timeout = task.get('timeout', self.config['task_timeout'])
                            
                            if elapsed > timeout:
                                # Task probably failed, reschedule
                                self.logger.info(f"Rescheduling task {task_id} that was running for {elapsed} seconds")
                                
                                # Update status in database
                                self._update_task_in_db(task_id, {
                                    'status': 'failed',
                                    'error': f"Task interrupted after {elapsed} seconds",
                                    'end_time': datetime.now()
                                })
                            else:
                                # Task might still be running, skip
                                self.logger.info(f"Skipping task {task_id} that might still be running ({elapsed} seconds)")
                                continue
                    
                    # Add to task queue
                    with self.task_lock:
                        # Store in task map
                        self.task_map[task_id] = new_task
                        
                        # Add to priority queue
                        heapq.heappush(self.tasks, (scheduled_time, priority, task_id, new_task))
                    
                    self.logger.info(f"Recovered task {task_id} ({name}) scheduled for {scheduled_time}")
                    
                except Exception as e:
                    self.logger.error(f"Error recovering task: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error recovering tasks: {e}")

    def _get_function_registry(self):
        """
        Get registry of available functions for task recovery.
        
        Returns:
            dict: Dictionary mapping function names to functions
        """
        # Default registry with common modules
        registry = {}
        
        # Add functions from this module
        import sys
        current_module = sys.modules[__name__]
        
        for name in dir(current_module):
            obj = getattr(current_module, name)
            if callable(obj):
                registry[name] = obj
                
        # Add functions from automation modules
        try:
            # Try to import automation modules
            from automation import daily_workflow, weekly_workflow, monthly_workflow
            
            # Add functions from daily workflow
            for name in dir(daily_workflow):
                obj = getattr(daily_workflow, name)
                if callable(obj):
                    registry[name] = obj
                    
            # Add functions from weekly workflow
            for name in dir(weekly_workflow):
                obj = getattr(weekly_workflow, name)
                if callable(obj):
                    registry[name] = obj
                    
            # Add functions from monthly workflow
            for name in dir(monthly_workflow):
                obj = getattr(monthly_workflow, name)
                if callable(obj):
                    registry[name] = obj
                    
        except ImportError:
            self.logger.warning("Could not import automation modules for function registry")
            
        return registry