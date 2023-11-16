'''
    The implement of the qualified exam module
'''

import os
import yaml
import warnings
import json
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from PRE.data import DataLoader
from PRE.api import Auto_API
from PRE.utils import parse_response

class EXAM:
    '''
    Conduct qualified exam, filtering qualified LLMs to become peer reviewers
    '''
    def __init__(self, args) -> None:
        self.source = args['source'] # same or others; same: the evaluated task and responses, others: independent prompts, no need for refer item
        self.mode = args['mode'] # pointwise, pairwise
        self.parser_type = args['parser_type'] # int, float, str
        '''
        If the source is same,
        In pointwise mode, the data consists key "#index" (the line index of the task) and key "#source" (the LLM to generate the response). The expected evaulate response is an integer or float number;
        In pairwise mode, the data consists key "#index" (the line index of the task), key "#source1" (the LLM 1 to generate the response) and key "#source2" (the LLM 2 to generate the response). The expected evaluate response is three possible token, meaning -1 (1 is better), 0 (tied), 1 (2 is better) respectively
        also, if we conduct reference exam, for each exam data item, it requires key "#answer" denotes the gold standard (integer for the pairwise mode)
        '''
        assert self.source in ['same', 'others']
        assert self.mode in ['pointwise', 'pairwise']
        assert self.parser_type in ['int', 'float', 'str']
        if self.parser_type == 'str':
            self.nominal_list = [nn.strip() for nn in args['nominal_list'].split(',')]
            self.nominal_ticks = [int(nn.strip()) for nn in args['nominal_list'].split(',')]
        else:
            self.nominal_list, self.nominal_ticks = None, None
        
        if self.source == 'same': # load generated task data and responses
            path_config_task_data = args['config_task_data']
            self.task_name = args['task_name']
            self.save_dir = args['save_dir'] # the exam result save dir, the exam evaluation save filename = [save_dir] / exam_responses / [task_name]_[model_name].json, each line is one result with json {response: str, result: float/int}
            if not os.path.exists(path_config_task_data):
                raise FileExistsError("Load task_data config failed: file not exist!")

            config_task = yaml.load(open(path_config_task_data, 'r'), Loader=yaml.FullLoader) # single task config
            data_loader = DataLoader(config_task) # a task data loader
            self.task_data = data_loader.get_task_items()
            self.path_exam_same_data = args['path_exam_same_data']
            self.format_exam_same_data = args['format_exam_same_data']
        else: # load other exam data
            self.path_exam_others_data = args['path_exam_others_data']
            self.format_exam_others_data = args['format_exam_others_data']
            if not os.path.exists(self.path_exam_others_data):
                raise FileExistsError("Load exam others mode data failed: file not exist!")
        self.reference_exam = args['conduct_reference_exam'] # True or False, whether to compare the responses v.s. gold standard
        self.inner_consistency_exam = args['conduct_inner_consistency_exam'] # True or False, whether to conduct inner-consistency exam
        if self.mode == 'pairwise':
            if self.reference_exam:
                self.p_gold = float(args['p_gold']) if 'p_gold' in args else 0.6 # accuarcy v.s. gold standard
            if self.inner_consistency_exam:
                self.p_cons = float(args['p_cons']) if 'p_cons' in args else 0.6 # consistency between two kinds of prompts
        elif self.mode == 'pointwise':
            self.metric_pointwise = args['metric_pointwise'] if 'metric_pointwise' in args else 'EM' # EM (exact match, proportion  >= threshold) or MSE (mean square error, mse <= threshold)
            assert self.metric_pointwise in ['EM', "MSE"]
            if self.reference_exam:
                if self.metric_pointwise == 'EM':
                    self.p_gold = float(args['p_gold']) if 'p_gold' in args else 0.6 # accuarcy v.s. gold standard
                elif self.metric_pointwise == 'MSE':
                    self.MSE_acc = float(args['MSE_gold']) if 'MSE_gold' in args else 1. # MSE v.s. gold standard
                
            if self.inner_consistency_exam:
                if self.metric_pointwise == 'EM':
                    self.p_cons = float(args['p_cons']) if 'p_cons' in args else 0.6 # consistency between two kinds of prompts
                elif self.metric_pointwise == 'MSE':
                    self.MSE_cons = float(args['MSE_cons']) if 'MSE_cons' in args else 1. # MSE between two kinds of prompts

        path_prompt = args['path_exam_prompt']
        if not os.path.exists(path_prompt):
            raise FileExistsError("Load exam prompt template failed: file not exist!")
        self.template_prompt = open(path_prompt, encoding='utf-8').read().strip()
        if self.inner_consistency_exam:
            path_prompt2 = args['path_exam_prompt2'] # used in inner consistency exam
            if not os.path.exists(path_prompt2):
                raise FileExistsError("Load exam prompt template 2 (used in inner-consistency exam) failed: file not exist!")
            self.template_prompt2 = open(path_prompt2, encoding='utf-8').read().strip()
        
        if not self.inner_consistency_exam and not self.reference_exam:
            warnings.warn("Have not set any qualified exam!", RuntimeWarning)
    
    
    def load_exam_prompts(self, prompt_template):
        if self.source == 'others':
            loader = DataLoader({"path_data": self.path_exam_others_data,
                                 "format": self.format_exam_others_data,})
            data_others = loader.get_task_items()
            prompts = []
            for item in data_others:
                prompt = prompt_template
                for key in item:
                    prompt = prompt.replace("{{" + key + "}}", item[key])
                prompts.append(prompt)
            if self.reference_exam:
                answers = [item['#answer'] for item in data_others]
            else:
                answers = None
            return prompts, answers
        elif self.source == 'same':
            loader = DataLoader({"path_data": self.path_exam_same_data,
                                 "format": self.format_exam_same_data,})
            samples_same = loader.get_task_items()
            evaluatees_list = set()
            if self.mode == 'pointwise':
                for sample in samples_same:
                    evaluatees_list.add(sample['#source'])
            elif self.mode == 'pairwise':
                for sample in samples_same:
                    evaluatees_list.add(sample['#source1'])
                    evaluatees_list.add(sample['#source2'])
            responses_evaluatee_dict = dict()
            for ev in evaluatees_list:
                responses = [] # responses list for evaluatee ev
                path = f"{self.save_dir}/task_responses/{self.task_name}_{ev}.json"
                if not os.path.exists(path):
                    raise FileExistsError(f"Load {path} failed: file not exist!")
                with open(path, 'r') as f:
                    while True:
                        line = f.readline().strip()
                        if line:
                            response = json.loads(line)['response']
                            responses.append(response)
                        else:
                            break
                responses_evaluatee_dict[ev] = responses
            
            prompts = []
            for sample in samples_same:
                sidx = sample['#index']
                task = dict(self.task_data[sidx])
                if self.mode == 'pointwise':
                    src = sample['#source']
                    task['#source'] = responses_evaluatee_dict[src][sidx]
                elif self.mode == 'pairwise':
                    src1 = sample['#source1']
                    src2 = sample['#source2']
                    task['#source1'] = responses_evaluatee_dict[src1][sidx]
                    task['#source2'] = responses_evaluatee_dict[src2][sidx]
                prompt = prompt_template
                for key in task:
                    prompt = prompt.replace("{{" + key + "}}", task[key])
                prompts.append(prompt)
            
            if self.reference_exam:
                answers = [item['#answer'] for item in samples_same]
            else:
                answers = None
            return prompts, answers
    
    def calculate_metric(self, resultsA, resultsB) -> float: 
        '''
        Calculate the evaluation metric between resultsA and resultsB
        pointwise or pairwise; EM/accuary or MSE (minus)
        '''
        assert len(resultsA) == len(resultsB)
        assert len(resultsA) > 0
        N = len(resultsA)
        p = 0.
        if self.mode == 'pairwise':
            for j in range(N):
                r, a = resultsA[j], resultsB[j]
                if r * a > 0:
                    p += 1.
                elif r * a == 0:
                    p += .5
            
        elif self.mode == 'pointwise':
            if self.metric_pointwise == 'EM':
                for j in range(N):
                    r, a = resultsA[j], resultsB[j]
                    if r == a:
                        p += 1.
            elif self.metric_pointwise == 'MSE':
                for j in range(N):
                    r, a = resultsA[j], resultsB[j]
                    p -= (r - a) ** 2

        p /= float(N)
        return p
        
        
    def conduct_exam(self, config_api_evaluator):
        '''
        Conduct qualified exam, return a list of qualified apis with the same format of list config_api_evaluator, and their scores [score_list (refer acc, inner acc) for each qualified LLM], MSE will put the minus one
        '''
        apis = [Auto_API.instantiate_api(config_api['api_type'], config_api) for config_api in config_api_evaluator]
        if not self.inner_consistency_exam and not self.reference_exam:
            return config_api_evaluator, [[] for _ in config_api_evaluator]
        
        prompts, answers = self.load_exam_prompts(self.template_prompt)
        if self.inner_consistency_exam:
            prompts2, answers2 = self.load_exam_prompts(self.template_prompt2)
        
        os.makedirs(f"{self.save_dir}/exam_responses", exist_ok=True)
        qualified_apis, scores_qualified = [], [] # configs of these qualified apis, its corresponding api
        for i, api in enumerate(apis):
            path_out = f"{self.save_dir}/exam_responses/{self.task_name}_{api.model_name}.json"

            if os.path.exists(path_out):
                data = open(path_out).readlines()
            else:
                data = []
            if len(data) < len(prompts):
                fout = open(path_out, 'w')
                for line in data:
                    fout.write(line)
                for prompt in prompts[len(data):]:
                    response_orig = api.chat(prompt)
                    result_parse = parse_response(response_orig, self.parser_type, self.nominal_list, self.nominal_ticks)
                    line = json.dumps({"response": response_orig,
                                        'result': result_parse})
                    data.append(line)
                    fout.write(line + '\n')
                fout.close()
            results = [json.loads(line.strip())['result'] for line in data]
            
            eval_this = [config_api_evaluator[i]]
            
            if self.reference_exam:
                p_refer = self.calculate_metric(results, answers)
                p_thre = None
                if self.mode == 'pairwise':
                    p_thre = self.p_gold
                elif self.mode == 'pointwise':
                    if self.metric_pointwise == 'EM':
                        p_thre = self.p_gold
                    elif self.metric_pointwise == 'MSE':
                        p_thre = -self.MSE_acc
                
                if p_refer < p_thre:
                    print(f'model {api.model_name} failed to pass the reference exam')
                    continue
                eval_this.append(p_refer)
            
            if self.inner_consistency_exam:
                path_out = f"{self.save_dir}/exam_responses/{self.task_name}_{api.model_name}__prompt2.json"

                if os.path.exists(path_out):
                    data = open(path_out).readlines()
                else:
                    data = []
                if len(data) < len(prompts2):
                    fout = open(path_out, 'w')
                    for line in data:
                        fout.write(line)
                    for prompt in prompts2[len(data):]:
                        response_orig = api.chat(prompt)
                        result_parse = parse_response(response_orig, self.parser_type, self.nominal_list, self.nominal_ticks)
                        line = json.dumps({"response": response_orig,
                                            'result': result_parse})
                        data.append(line)
                        fout.write(line + '\n')
                    fout.close()
                results2 = [json.loads(line.strip())['result'] for line in data]

                p_inner = self.calculate_metric(results, results2)
                p_thre = None
                if self.mode == 'pairwise':
                    p_thre = self.p_cons
                elif self.mode == 'pointwise':
                    if self.metric_pointwise == 'EM':
                        p_thre = self.p_cons
                    elif self.metric_pointwise == 'MSE':
                        p_thre = -self.MSE_cons
                
                if p_inner < p_thre:
                    print(f'model {api.model_name} failed to pass the inner-consistency exam')
                    continue
                eval_this.append(p_inner)
            
            qualified_apis.append(config_api_evaluator[i])
            scores_qualified.append(eval_this)
        return qualified_apis, scores_qualified
            
            

                    
            
            
        
        

        