import sys
from churn_prediction.logging import logger

class TelecomChurnException(Exception):
    def __init__(self, error_message,error_details:sys):
        #stores provided error message as instance variable
        self.error_message=error_message
        _,_,exc_tb=error_details.exc_info()

        #stores the exact line no where exception occured
        self.lineno=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename
    
    def __str__(self):
        return f"Error occure in python script name {self.file_name} line number {self.lineno} error message {str(self.error_message)}"

# if __name__=="__main__":
#     try:
#         logger.logging.info("Entered try block")
#         a=1/0
#         print("This will not be printed",a)
#     except Exception as e:
#         raise TelecomChurnException(e,sys)