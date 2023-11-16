'''
    The procedure of the whole peer review framework
'''

import os
import yaml
import json, csv
import copy

import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from PRE.data import DataLoader
from PRE.api import Auto_API
from PRE.exam import EXAM
from PRE.eval import PRE

class Process:
    '''
    The control of the whole peer review process
    '''
    @staticmethod
    def run(args): # the API used for automatic evaluation
        Process.collect_task_response(args)
        qualified_apis, scores_qualified = Process.conduct_qualified_exam(args)
        args['config_evaluators'] = qualified_apis
        args['scores_evaluators'] = scores_qualified
        # print(scores_qualified)
        Process.peer_review_and_evaluate(args)
        return None

    @staticmethod
    def collect_task_response(args):
        path_config_api_evaluatee = args['config_api_evaluatee']
        path_config_task_data = args['config_task_data']
        task_name = args['task_name']
        save_dir = args['save_dir'] # the task result save dir, the task save filename = [save_dir] / task_responses / [task_name]_[model_name].json, each line is one result with json {response: str}
        os.makedirs(os.path.join(save_dir, "task_responses"), exist_ok=True)
        
        if not os.path.exists(path_config_api_evaluatee):
            raise FileExistsError("Load api_evaluatee config failed: file not exist!")
        if not os.path.exists(path_config_task_data):
            raise FileExistsError("Load task_data config failed: file not exist!")
        
        config_apis = yaml.load_all(open(path_config_api_evaluatee, 'r'), Loader=yaml.FullLoader) # series of APIs
        config_task = yaml.load(open(path_config_task_data, 'r'), Loader=yaml.FullLoader) # single task config
        process_num = args['process_num'] # multi-process or not
        
        data_loader = DataLoader(config_task) # a task data loader
        apis = [Auto_API.instantiate_api(config_api['api_type'], config_api) for config_api in config_apis] # store for all valid apis
        prompts = [prompt for prompt in data_loader.get_prompt()]
        if args['debug']:
            print(prompts)
        for api in apis:
            path_out = f"{save_dir}/task_responses/{task_name}_{api.model_name}.json"
            if os.path.exists(path_out):
                responses = open(path_out).readlines()
            else:
                responses = []
            fout = open(path_out, 'w')
            for line in responses:
                fout.write(line)
            for prompt in prompts[len(responses):]:
                response = api.chat(prompt)
                fout.write(json.dumps({"response": response}) + '\n')
    
    @staticmethod
    def conduct_qualified_exam(args):
        path_config_exam = args['config_exam']
        if not os.path.exists(path_config_exam):
            raise FileExistsError("Load exam config failed: file not exist!")
        config_exam = yaml.load(open(path_config_exam, 'r'), Loader=yaml.FullLoader) # exam config
        args = copy.deepcopy(args)
        args.update(config_exam)
        exam = EXAM(args)
        path_config_api_evaluator = args['config_api_evaluator']
        config_apis = yaml.load_all(open(path_config_api_evaluator, 'r'), Loader=yaml.FullLoader) # series of APIs
        return exam.conduct_exam([cf for cf in config_apis])

    @staticmethod
    def peer_review_and_evaluate(args):
        pre = PRE(args)
        pre.evaluate()
