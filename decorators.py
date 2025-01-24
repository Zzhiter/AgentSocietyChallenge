from functools import wraps

from apt.utils.llm_logger import llm_logger

def log_llm_interaction(agent_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Access the parameters
            system_prompt = kwargs.get('system_prompt')
            user_input = kwargs.get('user_input')
            
            print("System Prompt:", system_prompt, "\nUser Input:", user_input, "\n")
            if system_prompt and user_input:
                # Log user input
                llm_logger.log_input(agent_name, user_input)
            
            # Execute the function
            full_response = func(self, *args, **kwargs)
            
            if full_response:
                # Log LLM response
                llm_logger.log_response(agent_name, full_response)
            
            return full_response
        return wrapper
    return decorator