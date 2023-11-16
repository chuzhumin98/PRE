'''
    The implement of various LLM APIs
'''

from abc import ABC, abstractmethod
import traceback, time
import requests
import json

import openai
import zhipuai

import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

class LLM_API(ABC):
    '''
    The base class for all kinds of LLM APIs
    '''
    @abstractmethod
    def __init__(self, args) -> None:
        self.model_name = args['model_name'] # model name
        self.api_type = None
    
    @abstractmethod
    def chat(self, prompt) -> str:
        '''
        The unified method for chatting with LLM, feeding the prompt, and then output the response
        '''
        raise NotImplementedError
    
    @staticmethod
    def instantiate_api(config):
        '''
        instantiate an LLM_API subclass object
        '''
        raise NotImplementedError


class OPENAI_API(LLM_API):
    '''
    OpenAI format API, please refer https://platform.openai.com/docs/api-reference/introduction for more information
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "openai"
        if 'api_key' not in args:
            raise Exception("Exception: missing openai API argument api_key")
        self.api_key = args['api_key']
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
        
    
    def chat(self, prompt) -> str:
        openai.api_key = self.api_key
        for _ in range(self.max_tries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                )
                if response:
                    if 'choices' in response:
                        answer = response['choices'][0]['message']['content'].strip()
                        return answer
                    else:
                        return response['error']['message']
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None

class CLAUDE_API(LLM_API):
    '''
    Claude-1 API with slack engine, we adopt claude2openai package for implement. please refer https://github.com/ChuxiJ/claude2openai for more information
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "claude"
        self.api_key = args['api_key']
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
        self.MAX_LEN = int(args['MAX_LEN']) if 'MAX_LEN' in args else 3960
        if 'slack_api_token' not in args:
            raise Exception("Exception: missing Claude API argument slack_api_token")
        if 'bot_id' not in args:
            raise Exception("Exception: missing Claude API argument bot_id")
        if 'channel_id' not in args:
            raise Exception("Exception: missing Claude API argument channel_id")
        self.slack_api_token = args['slack_api_token']
        self.bot_id = args['bot_id']
        self.channel_id = args['channel_id']
    
    def chat(self, prompt) -> str:
        import claude2openai
        claude2openai.slack_api_token = self.slack_api_token
        claude2openai.bot_id = self.bot_id
        claude2openai.channel_id = self.channel_id
        for _ in range(self.max_tries):
            try:
                prompt = prompt[:self.MAX_LEN]
                chat_completion = claude2openai.ChatCompletion.create(model="claude", messages=[{"role": "user", "content": prompt}])
                return chat_completion.choices[0].message.content
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None


class BAIDU_API(LLM_API):
    '''
    Baidu API with slack engine, please refer https://cloud.baidu.com/doc/WENXINWORKSHOP/s/flfmc9do2 for more information
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "baidu"
        if 'token' not in args:
            raise Exception("Exception: missing Baidu API argument token")
        self.token = args['token']
        self.url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.model_name}?access_token={self.token}"
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
    
    def chat(self, prompt) -> str:
        for _ in range(self.max_tries):
            try:
                payload = json.dumps({
                    "messages": [
                        {"role": "user", "content": prompt[:4800]}
                    ]
                })
                headers = {
                    'Content-Type': 'application/json'
                }

                response = requests.request("POST", self.url, headers=headers, data=payload)
                result = json.loads(response.text)['result']
                return result
            except:
                traceback.print_exc()
                time.sleep(5)
                continue
        return None


class GLM_API(LLM_API):
    '''
    GLM API with slack engine, please refer https://open.bigmodel.cn/dev/api for more information
    '''
    def __init__(self, args) -> None:
        self.model_name = args['model_name']
        self.api_type = "glm"
        if 'api_key' not in args:
            raise Exception("Exception: missing GLM API argument token")
        self.api_key = args['api_key']
        self.max_tries = int(args['max_tries']) if 'max_tries' in args else 5
    
    
    def chat(self, prompt) -> str:
        for _ in range(self.max_tries):
            try:
                zhipuai.api_key = self.api_key
                # request model
                response = zhipuai.model_api.invoke(
                    model=self.model_name,
                    prompt=[
                        {"role":"user", "content":prompt},
                    ]
                )
                print(response)
                if response['success']:
                    if 'choices' not in response['data']:
                        result = response['data']['outputText']
                    else:
                        result = response['data']['choices'][0]['content'].strip('"').strip().replace('\n', '\\n')
                else:
                    result = response['msg']
                return result
            except:
                traceback.print_exc()
                time.sleep(8)
                continue
        return None


# each item: [api_type, LLM_API child class name]
API_type2class_list = [['openai', OPENAI_API], ['claude', CLAUDE_API],
                       ['baidu', BAIDU_API], ['glm', GLM_API], ] 

class Auto_API:
    @staticmethod
    def instantiate_api(api_type, args) -> LLM_API:
        for at, _API in API_type2class_list:
            if api_type == at:
                return _API(args)
        raise Exception(f"Invalid api_type: {api_type}")