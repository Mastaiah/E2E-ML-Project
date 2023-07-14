import sys

def error_message_detail(message):
# sys.exc_info  functom will return exc_type, exc_value and exc_traceback
    _,_,exc_tb = sys.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = f"The exception has occured in python file {filename} lineno {exc_tb.tb_lineno} message {str(message)}"
    return error_message


#Custom exception class
class CustomException(message):
    def __init__(self,message):
        super().__init__(self)
        self.message = error_message_detail(message)

    def __str__(self):
        return self.message
    
