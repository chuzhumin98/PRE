'''
    The implement of the peer review and result aggregation module
'''

import os
import yaml
import warnings
import json
import copy
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from PRE.data import DataLoader
from PRE.api import Auto_API
from PRE.utils import parse_response
import numpy as np

class PEER_REVIEW:
    '''
    Conduct peer review, process for one prompt (pairwise or pointwise)
    '''
    def __init__(self, args) -> None:
        self.parser_type = args['parser_type'] # int, float, str
        self.task_name = args['task_name']
        self.save_dir = args['save_dir']
        if self.parser_type == 'str':
            self.nominal_list = [nn.strip() for nn in args['nominal_list'].split(',')]
            self.nominal_ticks = [int(nn.strip()) for nn in args['nominal_ticks'].split(',')]
        else:
            self.nominal_list, self.nominal_ticks = None, None
        
        
    def peer_review_single_round(self, reviewers, prompts):
        '''
        used in gaming sampling strategy
        reviewers: LLM config list
        prompts: an array, each item is a dict with key "prompt"
        return a dict to denote the results of each evaluate task under all the reviews, key: reviewer model name, value: the original response of this reviewer
        '''
        apis_reviewer = [Auto_API.instantiate_api(config_api['api_type'], config_api) for config_api in reviewers]
        responses_dict = dict()
        for _, api in enumerate(apis_reviewer):
            records_thisapi = []
            for prompt in prompts:
                response = api.chat(prompt['prompt'])
                result = parse_response(response, self.parser_type, self.nominal_list, self.nominal_ticks)
                item = {"response": response, "result": result}
                item.update(prompt)
                records_thisapi.append(item)
            responses_dict[api.model_name] = records_thisapi
        return responses_dict
    
    def peer_review_batch(self, reviewers, prompts) -> None:
        '''
        used in full evaluate strategy
        reviewers: LLM config list
        save the evaluate responses of each reviewer on seperated file
        '''
        apis_reviewer = [Auto_API.instantiate_api(config_api['api_type'], config_api) for config_api in reviewers]
        os.makedirs(f"{self.save_dir}/evaluation_responses", exist_ok=True)
        for _, api in enumerate(apis_reviewer):
            path_out = f"{self.save_dir}/evaluation_responses/{self.task_name}_{api.model_name}.json"

            if os.path.exists(path_out):
                data = open(path_out).readlines()
            else:
                data = []
            if len(data) < len(prompts):
                fout = open(path_out, 'w')
                for line in data:
                    fout.write(line)
                for prompt in prompts[len(data):]:
                    response_orig = api.chat(prompt['prompt'])
                    result_parse = parse_response(response_orig, self.parser_type, self.nominal_list, self.nominal_ticks)
                    line = {"response": response_orig, 'result': result_parse}
                    line.update(prompt)
                    line = json.dumps(line)
                    data.append(line)
                    fout.write(line + '\n')
                fout.close()        
        return None
 
 
class EvalDataLoader:
    def __init__(self, args) -> None:
        self.task_name = args['task_name']
        self.mode = args['mode'] # pointwise, pairwise
        '''
        In pointwise mode, the prompt is required to include key "#source" (the LLM to generate the response). The expected evaulate response is an integer or float number;
        In pairwise mode, the prompt is required to include key "#source1" (the LLM 1 to generate the response) and key "#source2" (the LLM 2 to generate the response). The expected evaluate response is three possible token, meaning -1 (1 is better), 0 (tied), 1 (2 is better) respectively
        '''
        # self.dirpath = args['dirpath_response'] # the load path for the response results
        self.save_dir = args['save_dir'] # the evaluation result save dir, In full strategy, the evaluation save filename = [save_dir] / evaluation_responses / [task_name]_[model_name].json, each line is one result with json {modelA: modelA_name, modelB: modelB_name, task_id: task_id, response: str, result: int/float}; in gaming strategy, the evaluation save filename = [save_dir] / evaluation_responses / [task_name]__[game strategy].json, each line is one compete result with json  {modelA: modelA_name, modelB: modelB_name, task_ids: list, response: {reviewer_name: {responses: list, results: list} for each reviewer}}
        self.path_evaluate_prompt = args['path_evaluate_prompt'] if 'path_evaluate_prompt' in args else None # the path of evaluate prompt template. In the prompt template, using {{key}} for the replacement of the key. For example, in the prompt "You need answer a question: {{question}}", the "question" field need to be included in the data 
    
        ### load task data and response data
        path_config_task_data = args['config_task_data']
        if not os.path.exists(path_config_task_data):
            raise FileExistsError("Load task_data config failed: file not exist!")
        config_task = yaml.load(open(path_config_task_data, 'r'), Loader=yaml.FullLoader) # single task config
        data_loader = DataLoader(config_task) # a task data loader
        self.task_data = data_loader.get_task_items()
        
        path_prompt = args['path_eval_prompt']
        if not os.path.exists(path_prompt):
            raise FileExistsError("Load evaluation prompt template failed: file not exist!")
        self.template_prompt = open(path_prompt, encoding='utf-8').read().strip()
        
        if 'config_api_evaluatee' in args:
            config_apis = yaml.load_all(open(args['config_api_evaluatee'], 'r'), Loader=yaml.FullLoader) # series of APIs
            self.evaluatee_LLM_names = [config_api['model_name'] for config_api in config_apis]
        else:
            self.evaluatee_LLM_names = args['evaluatee_names'].split(',')
        self.task_responses_dict = dict()
        self.N_task = None
        for ev in self.evaluatee_LLM_names:
            path = f"{self.save_dir}/task_responses/{self.task_name}_{ev}.json"
            if not os.path.exists(path):
                raise FileExistsError(f"Load {path} failed: file not exist!")
            responses = [] # responses list for evaluatee ev
            with open(path, 'r') as f:
                while True:
                    line = f.readline().strip()
                    if line:
                        response = json.loads(line)['response']
                        responses.append(response)
                    else:
                        break
            self.task_responses_dict[ev] = responses
            self.N_task = len(responses)

    
    def sample_pairwise_prompts(self, modelA, modelB, sample_size=5):
        '''
        sample some tasks between the comparison of modelA and modelB, with a given sample_size (used in ELO ranking system), each is a prompt
        '''
        assert modelA in self.task_responses_dict
        assert modelB in self.task_responses_dict
        responsesA, responsesB = self.task_responses_dict[modelA], self.task_responses_dict[modelB]
        idxs = range(len(self.N_task))
        np.random.shuffle(idxs)
        sample_ids = list(idxs)[:sample_size]
        prompts_list = [] # pairwise, then any template with two prompts
        for idx in sample_ids:
            task = dict(self.task_data[idx])
            prompt = self.template_prompt
            for key in task:
                prompt = prompt.replace("{{" + key + "}}", task[key])
            prompt_AB = prompt.replace('#source1', responsesA[idx]).replace("#source2", responsesB[idx])
            prompt_BA = prompt.replace('#source1', responsesB[idx]).replace("#source2", responsesA[idx])
            prompts_list += [prompt_AB, prompt_BA]
        return prompts_list, sample_ids

    def get_full_prompts(self):
        '''
        return all the prompts, under pointwise mode (K * N) or pairwise mode (K * (K-1) * N), each is a dict with key: model / modelA & modelB, prompt
        '''
        prompts_list = [] # each item is a dict {modelA: modelA_name, modelB: modelB_name, task_id: task_id, prompt: prompt} or {model: model_name, task_id: task_id, prompt: prompt}
        if self.mode == 'pointwise':
            for i in range(len(self.evaluatee_LLM_names)):
                for j in range(len(self.task_data)):
                    prompt = self.template_prompt
                    for key in self.task_data[j]:
                        prompt = prompt.replace("{{" + key + "}}", self.task_data[j][key])
                    prompt = prompt.replace('#source', self.task_responses_dict[self.evaluatee_LLM_names[i]][j])
                    prompts_list.append({'model': self.evaluatee_LLM_names[i], 'task_id': j, 'prompt': prompt})
        elif self.mode == 'pairwise':
            for i in range(len(self.evaluatee_LLM_names)):
                for j in range(i):
                    for k in range(len(self.task_data)):
                        modelA, modelB = self.evaluatee_LLM_names[i], self.evaluatee_LLM_names[j]
                        prompt = self.template_prompt
                        for key in self.task_data[k]:
                            prompt = prompt.replace("{{" + key + "}}", self.task_data[k][key])
                        prompt_AB = prompt.replace('#source1', self.task_responses_dict[modelA][k]).replace("#source2", self.task_responses_dict[modelB][k])
                        prompt_BA = prompt.replace('#source1', self.task_responses_dict[modelB][k]).replace("#source2", self.task_responses_dict[modelA][k])
                        prompts_list.append({'modelA': modelA, 'modelB': modelB, 'task_id': k, 'prompt': prompt_AB})
                        prompts_list.append({'modelA': modelB, 'modelB': modelA, 'task_id': k, 'prompt': prompt_BA})             
        return prompts_list

class PRE:
    def __init__(self, args) -> None:
        path_config_eval = args['config_eval']
        if not os.path.exists(path_config_eval):
            raise FileExistsError("Load config eval failed: file not exist!")
        args = copy.deepcopy(args)
        config_eval = yaml.load(open(path_config_eval, 'r'), Loader=yaml.FullLoader)
        args.update(config_eval)
        self.strategy = args['strategy'] # full, ELO, Glicko
        self.mode = args['mode'] # pointwise, pairwise
        if self.strategy in ['ELO', 'Glicko']:
            self.mode = 'pairwise' # sampling strategy, default with pairwise mode
            args['mode'] = 'pairwise'
        assert self.strategy in ['full', 'ELO', 'Glicko']
        assert self.mode in ['pointwise', 'pairwise']
        self.weighted_method = args['weighted_method'] # uniform, log (only accuary/consistency), exp, poly (only accuary/consistency)
        '''
        uniform: the equal weight
        log: log(p) - log(1-p)
        exp: exp(alpha * p)
        poly: p ^ alpha
        '''
        self.alpha = args['alpha'] if 'alpha' in args else 1.
        self.w_gold = args['w_gold'] if 'w_gold' in args else 0.5 # w_gold * s_gold + (1-w_gold) * s_consistency, only used when both of them are used in exam module
        self.evaluators_config = yaml.load_all(open(args['config_api_evaluator'], 'r'), Loader=yaml.FullLoader) # the config of evaluators
        self.evaluators_config = [cf for cf in self.evaluators_config]
        self.evaluator_model_names = [ev['model_name'] for ev in self.evaluators_config]
        self.save_dir = args['save_dir']
        self.task_name = args['task_name']
        # print(f"evaluatee config: {args['config_api_evaluatee']}")
        if 'config_api_evaluatee' in args:
            config_apis = yaml.load_all(open(args['config_api_evaluatee'], 'r'), Loader=yaml.FullLoader) # series of APIs
            self.evaluatee_LLM_names = [config_api['model_name'] for config_api in config_apis]
        else:
            self.evaluatee_LLM_names = args['evaluatee_names'].split(',')
        
        self.loader_data = EvalDataLoader(args)
        self.review = PEER_REVIEW(args)
        self.weights = self.weighted_function(args['scores_evaluators']) # the pre-compute weights of each evaluator based on their scores
        return
    
    def load_batch_data(self):
        prompts = self.loader_data.get_full_prompts()
        self.review.peer_review_batch(self.evaluators_config, prompts) # generate the peer review results of each evaluator
        ### load evaluation results
        results = dict()
        for ev_model_name in self.evaluator_model_names:
            path_ev = f"{self.save_dir}/evaluation_responses/{self.task_name}_{ev_model_name}.json"
            results_thisllm = []
            with open(path_ev, 'r') as f:
                while True:
                    line = f.readline().strip()
                    if line:
                        results_thisllm.append(json.loads(line))
                    else:
                        break
            results[ev_model_name] = results_thisllm
        return results
    
    def evaluate(self):
        '''
        the unified api for evaluate, control the whole evaluation procedure
        '''
        if self.strategy == 'full':
            self.evaluate_full()
        else:
            self.evaluate_sample()

    def evaluate_full(self):
        '''
        evaluate with the full strategy
        '''
        results = self.load_batch_data()
        ### evaluate with majority voting
        os.makedirs(f"{self.save_dir}/evaluation_results", exist_ok=True)
        print(self.evaluatee_LLM_names)
        if self.mode == 'pointwise':
            results_perllm = dict() # evaluate dict of each evaluatee
            for ev in self.evaluator_model_names:
                results_ev = results[ev]
                for item in results_ev:
                    model, task_id, label = item['model'], item['task_id'], item['result']
                    if model not in results_perllm:
                        results_perllm[model] = dict()
                    if task_id not in results_perllm[model]:
                        results_perllm[model][task_id] = []
                    results_perllm[model][task_id].append(label)
            outputs = dict()
            for model in results_perllm:
                outputs[model] = []
                for task_id in results_perllm[model]:
                    outputs[model].append(self.aggregate_reviewers_results(results_perllm[model][task_id], self.weights))
            path_res = f"{self.save_dir}/evaluation_results/{self.task_name}_result_detail.json"
            json.dump(outputs, open(path_res, 'w'))
            with open(f"{self.save_dir}/evaluation_results/{self.task_name}_result_overview.txt", 'w') as f:
                for model in outputs:
                    mean_val = np.mean(outputs[model])
                    print(f'model {model}: {mean_val}')
                    f.write(f'model {model}: {mean_val}\n')
        elif self.mode == 'pairwise':
            results_perllm = dict() # evaluate dict of each evaluatee
            for i, ev in enumerate(self.evaluator_model_names):
                results_ev = results[ev]
                for item in results_ev:
                    modelA, modelB, task_id, label = item['modelA'], item['modelB'], item['task_id'], item['result']
                    if modelA <= modelB:
                        key = f'{modelA}%{modelB}'
                    else:
                        key = f'{modelB}%{modelA}'
                        label = -label # reversed the preference label if modelB v.s. modelA
                    
                    if key not in results_perllm:
                        results_perllm[key] = dict()
                    if task_id not in results_perllm[key]:
                        results_perllm[key][task_id] = []
                    if len(results_perllm[key][task_id]) < i + 1:
                        results_perllm[key][task_id].append([])
                    results_perllm[key][task_id][i].append(label)
            outputs = dict()
            for key in results_perllm:
                outputs[key] = []
                for task_id in results_perllm[key]:
                    outputs[key].append(self.aggregate_reviewers_results(results_perllm[key][task_id], self.weights))
            path_res = f"{self.save_dir}/evaluation_results/{self.task_name}_result_detail.json"
            json.dump(outputs, open(path_res, 'w'))
            with open(f"{self.save_dir}/evaluation_results/{self.task_name}_result_overview.csv", 'w') as f:
                evaluatees_dict = {ev: i for i, ev in enumerate(self.evaluatee_LLM_names)}
                accs = np.zeros([len(self.evaluatee_LLM_names), len(self.evaluatee_LLM_names)], dtype=np.float)
                for key in outputs:
                    mA, mB = key.split('%')
                    idxA, idxB = evaluatees_dict[mA], evaluatees_dict[mB]
                    res = np.array(outputs[key])
                    mean_val = np.mean(res == 1) + np.mean(res == 0) * 0.5
                    accs[idxA, idxB] = mean_val
                    accs[idxB, idxA] = 1. - mean_val
                f.write(','.join(['']+self.evaluatee_LLM_names) + '\n')
                for i in range(len(self.evaluatee_LLM_names)):
                    f.write(','.join([self.evaluatee_LLM_names[i]] + [str(num) for num in accs[i]]) + '\n')
            lines = open(f"{self.save_dir}/evaluation_results/{self.task_name}_result_overview.csv", 'r').readlines()
            print(''.join(lines))
    
    def evaluate_sample(self):
        '''
        evaluate with sampling strategies (e.g. ELO, Glicko)
        '''
        results = self.load_batch_data()
        ### only for pairwise mode
        os.makedirs(f"{self.save_dir}/evaluation_results", exist_ok=True)
        results_perllm = dict() # evaluate dict of each evaluatee
        for i, ev in enumerate(self.evaluator_model_names):
            results_ev = results[ev]
            for item in results_ev:
                print(item)
                modelA, modelB, task_id, label = item['modelA'], item['modelB'], item['task_id'], item['result']
                if modelA <= modelB:
                    key = f'{modelA}%{modelB}'
                else:
                    key = f'{modelB}%{modelA}'
                    label = -label # reversed the preference label if modelB v.s. modelA
                
                if key not in results_perllm:
                    results_perllm[key] = dict()
                if task_id not in results_perllm[key]:
                    results_perllm[key][task_id] = []
                if len(results_perllm[key][task_id]) < i + 1:
                    results_perllm[key][task_id].append([])
                results_perllm[key][task_id][i].append(label)
        games_list = []
        evaluatees_dict = {ev: i for i, ev in enumerate(self.evaluatee_LLM_names)}

        for key in results_perllm:
            mA, mB = key.split('%')
            idxA, idxB = evaluatees_dict[mA], evaluatees_dict[mB]
            for task_id in results_perllm[key]:
                games_list.append([idxA, idxB, self.aggregate_reviewers_results(results_perllm[key][task_id], self.weights)])
        indexes = np.array(range(len(games_list)))
        np.random.shuffle(indexes) # randomize the game order
        path_res = f"{self.save_dir}/evaluation_results/{self.task_name}_result_detail.txt"
        fout = open(path_res, 'w')
        if self.strategy == 'ELO': # we set K = 16
            def elo_expect_win_rate(x): # x is the ELO difference
                return 1. / (1. + 10. ** (x / 400.))
            rates = [1000., 1000.]
            K = 16.
            for r, idx in enumerate(indexes):
                roleA, roleB, label = games_list[idx]
                eA = elo_expect_win_rate(rates[roleB] - rates[roleA])
                eB = 1. - eA
                sB = (1. + label) / 2. # -1 -> 0, 0 -> 0.5, 1 -> 1
                sA = 1. - sB
                rates[roleA] += K * (sA - eA)
                rates[roleB] += K * (sB - eB)
                fout.write(f"After round {r}, ELO rate: {rates}\n")
        elif self.strategy == 'Glicko':
            # TODO
            pass
        fout.close()
        
        with open(f"{self.save_dir}/evaluation_results/{self.task_name}_result_overview.csv", 'w') as f:
            f.write(f"Final {self.strategy} rate leaderboard:\n")
            ranks = np.argsort(-np.array(rates))
            for r in ranks:
                f.write(f"{self.evaluatee_LLM_names[r]}: {rates[r]}\n")
        lines = open(f"{self.save_dir}/evaluation_results/{self.task_name}_result_overview.csv", 'r').readlines()
        print(''.join(lines))
    
    def weighted_function(self, scores):
        '''
        return the weight (normalized) of each LLM, with the given weighted method and parameter (alpha and w_gold)
        '''
        assert len(scores) > 0
        N = len(scores)
        if len(scores[0]) == 0 or self.weighted_method == 'uniform': # when no exam or uniform strategy, equal weight
            p = 1. / float(N)
            return np.array([p for _ in range(N)])
        elif self.weighted_method == 'log':
            ws = np.log([s[0] for s in scores]) - np.log([1. - s[0] for s in scores])
            if len(scores[0]) > 1:
                ws2 = np.log([s[1] for s in scores]) - np.log([1. - s[1] for s in scores])
                ws = self.w_gold * ws + (1-self.w_gold) * ws2
            ws /= np.sum(ws)
            return ws
        elif self.weighted_method == 'exp':
            ws = np.exp(self.alpha * np.array([s[0] for s in scores]))
            if len(scores[0]) > 1:
                ws2 = np.exp(self.alpha * np.array([s[1] for s in scores]))
                ws = self.w_gold * ws + (1-self.w_gold) * ws2
            ws /= np.sum(ws)
            return ws
        elif self.weighted_method == 'poly':
            ws = np.array([s[0] for s in scores]) ** self.alpha
            if len(scores[0]) > 1:
                ws2 = np.array([s[1] for s in scores]) ** self.alpha
                ws = self.w_gold * ws + (1-self.w_gold) * ws2
            ws /= np.sum(ws)
            return ws
        else:
            raise Exception("Unexpected parameter weighted_method!")

    
    def aggregate_reviewers_results(self, results, weights):
        '''
        aggregate results with the given weights
        if mode == 'pointwise', results and weights are all (N) array, N is the size of evaluators; weighted sum
        if mode == 'pairwise', results are (N, 2) array, and weights are (N) array; majority voting, pairwise is already aligned, i.e., if B ~ A is better, then convert into A ~ B is worse
        '''
        assert len(results) == len(weights)
        if self.mode == 'pointwise':
            return sum([results[i] * weights[i]  for i in range(len(weights))])
        elif self.mode == 'pairwise':
            cnt_pos, cnt_neg = 0., 0.
            for items in results:
                for item in items:
                    if item > 0:
                        cnt_pos += 1.
                    elif item < 0:
                        cnt_neg += 1.
            if cnt_pos > cnt_neg:
                return 1
            elif cnt_pos < cnt_neg:
                return -1
            else:
                return 0