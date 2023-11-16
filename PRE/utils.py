'''
    Implement some commonly used functions
'''


import re
import math
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

def parse_response(response, parse_type, nominal_list=None, nominal_ticks=None):
    '''
    parse_type: int, float or str
    if parse_type = str, then required parameter nominal_list and nominal_ticks
    nominal_list: a series of nominal types, its name
    nomianl_ticks: the corresponding nominal number (int)
    '''
    assert parse_type in ['int', 'float', 'str']
    if parse_type == 'int':
        nums = re.findall(r"-?\d+", response)
        if len(nums) == 0:
            return None
        return int(nums[0])
    elif parse_type == 'float':
        nums = re.findall(r"-?\d+\.?\d*", response)
        if len(nums) == 0:
            return None
        return int(nums[0])
    elif parse_type == 'str':
        appear_pos, cur_idx = math.inf, -1
        response = response.lower()
        for idx, label in enumerate(nominal_list):
            pos = response.find(label.lower())
            if pos != -1: # really appear!
                if pos < appear_pos:
                    appear_pos, cur_idx = pos, idx
        if cur_idx == -1:
            return None
        else:
            return nominal_ticks[cur_idx]
    