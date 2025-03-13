from google import genai
import time
import re 
import pandas as pd

class LLMInterface:
    def __init__(self, api_token):
        self.api_token = api_token
        self.client = self.create_client(api_token)

    def create_client(self, api_token):
        """
        Given an API token, create an LLM model client for easy interaction with the model

        Parameters:
            api_token (str): API token for the LLM model
        Returns:
            client: LLM model client object
        """
        return

    def generate_content(self, prompt):
        """
        Given a prompt, generate content using the LLM model

        Parameters:
            prompt (str): Prompt for the LLM model
        Returns:
            response (str): Response from the LLM model
        """
        return "This would be the generated content"

class GeminiModel(LLMInterface):

    def __init__(self, api_token, requests_per_minute=15):
        self.requests_per_minute = requests_per_minute
        super().__init__(api_token)

    def create_client(self, api_token):
        """
        Given an API token, create a Gemini model client for easy interaction with the model

        Parameters:
            api_token (str): API token for the Gemini model
        Returns:
            client: Gemini model client object
        """
        return genai.Client(api_key=api_token)
    
    def generate_content(self, prompt, model_name="gemini-2.0-flash"):
        """
        Given a prompt, generate content using the Gemini model. Default model is gemini-2.0-flash
        
        WARNING: This method is rate limited to self.requests_per_minute requests per minute and calls time.sleep(). Be careful in multi-threaded environments.
        
        Parameters:
            prompt (str): Prompt for the Gemini model
            model_name (str): Name of the model to use
        Returns:
            response (str): Response from the Gemini model
        """
        request = self.client.models.generate_content(
            model=model_name, contents=prompt,
        )

        sleep_time = 60 / self.requests_per_minute * 1.1
        time.sleep(sleep_time)

        ### Check if it included markdown syntax and extract the code snippet
        response = request.text
        if "```cpp" in response:
            response = response.split("```cpp")[1].split("```")[0]

        ### remove beginning and trailing white space and newlines
        response = response.strip()
        response = response.strip("\n")
        
        return response 

def code_metrics(code) -> list[float]:
    """
    Calculate the number of lines, code length, comments, num of indents, loop count, avg line length, identifiers, readability

    Parameters:
        code (str): code snippet
    Returns:
        metrics (list): List of code metrics calculated from the code snippet (num_of_lines, code_length, comments, cyclomatic complexity, num_indents, loop_count, avg_line_length, identifiers, readability)

    """
    num_of_lines = len(code.split("\n"))
    code_length = len(code)
    comments = len(re.findall(r'//', code)) + (len(re.findall(r'/\*', code)) + len(re.findall(r'\*/', code))) // 2
    num_indents = len(re.findall(r'\t', code)) + len(re.findall(r'    ', code))
    loop_count = len(re.findall(r'for', code)) + len(re.findall(r'while', code))
    avg_line_length = code_length / num_of_lines
    cyclomatic_complexity = len(re.findall(r'if|else|for|while|case|default|continue|break|&&|\|\|', code))
    identifiers = len(re.findall(r'\b\w+\b', code)) / num_of_lines
    readability = 0.39 * avg_line_length + 0.1 * num_indents + 0.15 * comments + 0.2 * loop_count + 0.16 * identifiers

    return [num_of_lines,  code_length,  comments, cyclomatic_complexity, num_indents,  loop_count,  avg_line_length,  identifiers, readability]