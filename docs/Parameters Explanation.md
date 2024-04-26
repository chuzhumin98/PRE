# Parameters Explanation

In this file, we present the specific explanation for the parameters which is required to set in the .yaml config files.

### main.yaml

This yaml file is the root config file of the whole project, which points to the path of the yaml file for each submodule.

- config_api_evaluatee: the file path of the evaluatee LLMs config file (relative to the *main.py*)
- config_api_evaluator: the file path of the evaluator LLMs config file (relative to the *main.py*)
- config_task_data: the file path of the task data config file (relative to the *main.py*)
- config_exam: the file path of the qualification exam module config file (relative to the *main.py*)
- config_eval: the file path of the peer review and "chair" decision modules config file (relative to the *main.py*)
- task_name: the name of this batch of evaluation task, the task name is appended on the file names of evaluation results
- save_dir: the save directory path of the evaluation results

### evaluatees.yaml / evaluators.yaml

These two file consist a list of LLMs' config items, each item is seperated by line "---".  For each LLM's config item, model_name and api_type are two required fields, others are customed by particular LLM api type (please refer to PRE/api.py for the detailed fields).

- model_name: the evaluatee (or evaluator) LLM model name, used in the subsequent evaluation reports
- api_type: the api type of this evaluatee (or evaluator) LLM, which is binded with the specific API class type (the list *API_type2class_list* provides the specific mapping relation)

### data.yaml

This yaml file consists the config parameters used in the data preparation procedure (data.yaml).

- path_data: the file path of the raw task samples data
- format: the organization format of the task samples data, optional formats include csv, json
- path_prompt: the file path of the task prompt template

### exam.yaml

This yaml file consists the config parameters used in the qualification exam module (exam.yaml)

- source: the value is *same* or *others*. *same* denotes the evaluated task and responses are the same with the peer review module; *others* denotes the exam data is independent with the peer review module
- mode: the exam evaluation mode, optional formats include pointwise, pairwise
- parse_type: the parsed label type, optional formats include int, float, str

- nominal_list: the candidate nominal label list, labels are split with ",". This parameter is only used when parse_type is "str"
- conduct_reference_exam: boolean type, whether to conduct reference-based exam
- conduct_inner_consistency_exam: boolean type, whether to conduct inner-consistency-based exam
- metric_pointwise: the evaluation metric used in pointwise mode exam, optional metrics include EM (exact match, proportion  >= threshold) and MSE (mean square error, mse <= threshold). Note that in pairwise mode exam, only EM is acceptable

- p_gold: the EM metric threshold used in reference-based exam
- p_cons: the EM metric threshold used in inner-consistency-based exam
- MSE_gold: the MSE metric threshold used in reference-based exam
- MSE_cons: the MSE metric threshold used in inner-consistency-based exam

- path_exam_prompt: the file path of the exam prompt template
- path_exam_prompt2: the file path of the second type of exam prompt template (only used in inner-consistency-based exam)
- path_exam_same_data: the file path of the exam data, used in the same source
- format_exam_same_data: the organization format of the exam data, used in the same source, optional formats include csv, json
- path_exam_others_data: the file path of the exam data, used in the others source
- format_exam_others_data: the organization format of the exam data, used in the others source, optional formats include csv, json

### eval.yaml

This yaml file consists the config parameters used in the peer review and "chair" decision modules.

- strategy: the adopted evaluation strategy, candidate choices include *full*, *ELO* and *Glicko*. *full* means use all the rating results to obtain the final evaluation results; *ELO* and *Glicko* means use sampling strategy to sample a part of preference pair samples and the ELO and Glicko rating algorithm to obtain the final evaluation results
- mode: the evaluation mode, optional formats include pointwise, pairwise; in ELO and Glicko strategies, default mode is pairwise
- weighted_method: the weighted voting method, optional choices include uniform, log, exp and poly. uniform means the equivalent weight, log means $$\log [p / (1-p)]$$, exp means $$\exp(\alpha p)$$, poly means $$p^\alpha$$. Note that the $p$ is the accuracy obtained in qualification exam module (if both reference-based and inner-consistency-based exams are taken, then we will use w_gold parameter to aggregate these two metrics first to obtain $p$), so MSE metric is unacceptable in these two methods
- alpha: used when weighted_method is exp or poly, default value is 1
- w_gold: both reference-based and inner-consistency-based exams are taken, parameter w_gold is then used to aggregate these two accauray metrics. then s_total = w_gold * s_reference + (1 - w_gold) * s_consistency. Here w_gold is a real number range from 0 to 1
- path_eval_prompt: the file path of the evaluation prompt template
- parser_type: the parsed label type, optional formats include int, float, str
- nominal_list: the nominal labels list when parser_type is str
- nominal_ticks: the nominal ticks list corresponding to the nominal list 