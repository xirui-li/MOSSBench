import os
from MOSSBench.models import load_model
from MOSSBench.utils.utils import (
                                  setup_logger,
                                  OverDataset,
                                  fix_seed,
                                  append_to_jsonl,
                                  read_from_jsonl,
                                  load_keys,
                                  write_to_json,
                                  read_from_json)
from MOSSBench.evaluation.Evaluator import Evaluator
from datetime import datetime
import re
import numpy as np
from MOSSBench.evaluation.evaluation_prompts import EVAL

import argparse

def get_parser():
    """
    Setup the command line arguments.

    Returns:
        argparse.ArgumentParser: Returns the parser with the necessary arguments.
    """
    parser = argparse.ArgumentParser(description="Main Arguments")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument("--max_num_worker", type=int, default=1, help="maximum number of workers for dataloader")
    parser.add_argument("--eval_model_name", type=str,default="gpt-4-turbo")
    parser.add_argument("--model_name", type=str, default="gemini-1.5-pro-latest") 
    parser.add_argument("--model_weight_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--inference", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--data_offset", type=int, default=0)
    parser.add_argument("--data_list", nargs='+', help="Specify the data samples to be run")
    parser.add_argument("--result_dir", type=str, help="Continue from previous progress")
    parser.add_argument("--google_safety_level", type=str, default='BLOCK_MEDIUM_AND_ABOVE')

    return parser


def main():

    parser = get_parser()
    args = load_keys(parser.parse_args())

    fix_seed(args.seed)

    total_rejects = []
    type_rejects = [[], [], []]

    def add_per_type_reject(type_text, rejected):
        type = re.search(r'\d+', type_text)
        if type:
            type_rejects[int(type.group()) - 1].append(rejected)

    if "llava" in args.model_name:
        prefix = "llava"
    elif "instructblip" in args.model_name:
        prefix = "instructblip"
    elif "blip2" in args.model_name:
        prefix = "blip2"
    elif "Qwen" in args.model_name:
        prefix = "Qwen"
    elif "idefics" in args.model_name:
        prefix = "idefics"
    elif "cogvlm" in args.model_name:
        prefix = "cogvlm"
    elif "Yi-VL" in args.model_name:
        prefix = "Yi"
    elif "internlm" in args.model_name:
        prefix = "internlm"
    elif "InternVL" in args.model_name:
        prefix = "InternVL" 
    else:
        prefix = args.model_name

    if args.data_list:
        for pid in args.data_list:
            assert int(pid) >= args.data_offset, "Specified data pid must be greater than the offset"

    start_time = datetime.now().strftime('%m-%d %H-%M-%S')
    results = {}
    if not args.result_dir:
        result_dir = os.path.join("experiments/results", prefix + " " + start_time)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        results['responses'] = {}

    else:  # Continue from previous progress if result_dir is specified
        result_dir = args.result_dir
        results = read_from_json(os.path.join(result_dir, "results.json"))

        if args.inference:  # Only set up when resume to inference
            # Delete all responses that to be updated
            if args.data_list:
                for pid in args.data_list:
                    if pid in results['responses'].keys():
                        del results['responses'][pid]

            for pid, result in results['responses'].items():
                add_per_type_reject(result['metadata']['over'], result['rejected'])
                total_rejects.append(result['rejected'])

    logger = setup_logger(args, result_dir)

    bench = OverDataset(args)

    model = load_model(args)

    evaluator = Evaluator(args.eval_model_name, args.openai_api_key, EVAL)

    for i, data in enumerate(bench):

        if int(data['pid']) < args.data_offset:
            continue

        if args.inference and data['pid'] in results['responses']:
            continue

        if args.data_list and data['pid'] not in args.data_list:
            continue

        result = {
            "pid": data['pid'],
            "image_path": data['image'],
            "question": data['question'],
            "response": '',
            "rejected": None,
            "reject_reasoning": '',
            "metadata": data['metadata']
        }

        if args.inference:
            logger.info(f"Inference PID: {data['pid']}, Question: {data['question']}")
            response = model.generate(data['question'], data['image'])
            result['response'] = response
            results['responses'][data['pid']] = result

        if args.eval:
            if not result['rejected']:  # Re-evaluating so load result from results.json
                result = results['responses'][data['pid']]
            description, rejected, reject_reasoning = evaluator.eval(data['image'], data['question'], result['response'])

            result['rejected'] = rejected
            result['reject_reasoning'] = reject_reasoning
            result['evaluator_desc'] = description

            total_rejects.append(rejected)

            # Per type reject rate
            add_per_type_reject(data['metadata']['over'], rejected)

            results['reject Rate'] = {
                "Total": np.mean(total_rejects),
                "Exaggerated Risk": np.mean(type_rejects[0]),
                "Negated Harm": np.mean(type_rejects[1]),
                "Counterintuitive Interpretation": np.mean(type_rejects[2])
            }

            logger.info(f"pid: {data['pid']} Rejected?: {rejected} Current Reject Rate: {np.mean(total_rejects)}")
            logger.info(f"[Exaggerated Risk] reject rate: {np.mean(type_rejects[0])}")
            logger.info(f"[Negated Harm] reject rate: {np.mean(type_rejects[1])}")
            logger.info(f"[Counterintuitive Interpretation] reject rate: {np.mean(type_rejects[2])}")

        if args.inference:
            write_to_json(results, os.path.join(result_dir, "results.json"))
        elif args.eval:
            write_to_json(results, os.path.join(result_dir, f"re_evaluate {start_time}.json"))


    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

if __name__ == "__main__":
    main()