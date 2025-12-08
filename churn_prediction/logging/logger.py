import logging
import os
from datetime import datetime

# 1. Generate timestamp ONCE
# datetime.now() gets current date and time, strftime() converts into string, timestamp stores this
# Eg: 12_08_2025_08_14_22
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# create a log file by appending .log to the timestamp
# Eg: 12_08_2025_08_14_22.log, this is f-string
LOG_FILE = f"{timestamp}.log"

# 2. Create logs directory only
# os.getcwd() gets cwd and os.path.join() joins cwd with logs 
# Eg: End to End Telecom Churn Prediction/logs
logs_dir = os.path.join(os.getcwd(), "logs")

# creates the logs directory from earlier path, exist_ok doesn't raise error if folder already exits
os.makedirs(logs_dir, exist_ok=True)

# 3. Full log file path
# combines logs direcotry path with LOG_FILE
#Eg: End to End Telecom Churn Prediction/logs/12_08_2025_08_14_22.log
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# 4. Configure logging

logging.basicConfig(
    # sends all log messages to LOG_FILE_PATH instead of console
    filename=LOG_FILE_PATH,
    # format decide how each log line looks like
    # 1. time of log record, 2. inserts lineno in source file where logging.info() called
    # 3. inserts logger's name 4. inserts logging level(INFO,ERROR) 5. inserts actual log message text
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    # sets  minimum severity level to INFO,ERROR,WARNING,ERROR,CRITICAL messages are logged
    # DEBUG messages are ignored
    level=logging.INFO,
)