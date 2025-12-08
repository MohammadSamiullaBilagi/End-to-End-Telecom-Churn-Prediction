import logging
import os
from datetime import datetime

# 1. Generate timestamp ONCE
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
LOG_FILE = f"{timestamp}.log"

# 2. Create logs directory only
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# 3. Full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# 4. Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)