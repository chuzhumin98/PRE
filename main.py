import argparse
import os
import yaml
import json, csv
import sys


from PRE.process import Process

def main(args):
    
    if not os.path.exists(args['config']):
        raise FileExistsError("Load config failed: file not exist!")
        
    args_config = yaml.load(open(args['config'], 'r'), Loader=yaml.FullLoader)
    Process.run(args_config)
    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate LLM by peer reviewing')
    parser.add_argument("--config")
    args = parser.parse_args()
    
    args = vars(args)
    main(args)