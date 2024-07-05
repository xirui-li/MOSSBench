import random
import numpy as np
import torch

from torch.utils.data import Dataset
import json
import os
import multiprocessing
from PIL import Image
import logging

def load_keys(args):
    ### Load API keys
    if os.path.exists("api_keys/google_keys.txt"):
        with open("api_keys/google_keys.txt") as f:
            google_key = f.read()
    else:
        google_key = ''

    if os.path.exists("api_keys/anthropic_keys.txt"):
        with open("api_keys/anthropic_keys.txt") as f:
            anthropic_key = f.read()
    else:
        anthropic_key = ''

    if os.path.exists("api_keys/reka_keys.txt"):
        with open("api_keys/reka_keys.txt") as f:
            reka_key = f.read()
    else:
        reka_key = ''

    if os.path.exists("api_keys/openai_keys.txt"):
        with open("api_keys/openai_keys.txt") as f:
            openai_keys = f.readlines()
            openai_api_key = openai_keys[0].strip()
            if len(openai_keys) > 1:
                openai_api_org = openai_keys[1]
            else:
                openai_api_org = ''
    else:
        openai_api_key = ''
        openai_api_org = ''

    args.google_key = google_key
    args.openai_api_key = openai_api_key
    args.openai_api_org = openai_api_org
    args.anthropic_key = anthropic_key
    args.reka_key = reka_key

    return args

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def append_to_jsonl(result, file_path):
    with open(file_path, 'a') as file:
        file.write(json.dumps(result) + '\n')

def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def write_to_json(result, file_path):
    with open(file_path, 'w') as file:
        json.dump(result, file, indent=4)

def read_from_json(file_path):
    with open(file_path, 'r') as file:
        result = json.load(file)
    return result

def setup_logger(args, result_dir):
    logger = logging.getLogger('experiment_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(result_dir, 'experiment.log'))
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("-" * 100)
    logger.info(f'Experiment Config: {args}')
    logger.info("-" * 100)

    return logger


class OverDataset(Dataset):
    def __init__(self, args, offset=0):
        super().__init__()
        with open(os.path.join(args.data_dir, "images_information/information.json")) as f:
            self.scenes = list(json.load(f).items())
        self.len = len(self.scenes)
        self.data_dir = args.data_dir
        self.offset = offset

    def __len__(self):
        return self.len - self.offset

    def __getitem__(self, index):
        scene = self.scenes[index + self.offset]
        k, v = scene
        v['image'] = os.path.join(self.data_dir, v['image'])
        return v
    
class HarmbenchDataset(Dataset):
    def __init__(self, args, offset=0):
        super().__init__()
        with open(os.path.join(args.data_dir, "harmbench_images_information/information.json")) as f:
            self.scenes = list(json.load(f).items())
        self.len = len(self.scenes)
        self.data_dir = args.data_dir
        self.offset = offset

    def __len__(self):
        return self.len - self.offset

    def __getitem__(self, index):
        scene = self.scenes[index + self.offset]
        k, v = scene
        v['image'] = os.path.join(self.data_dir, v['image'])
        # import pdb; pdb.set_trace()
        return v

def setup_data_loader(args):
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = HarmbenchDataset(args)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=True,
                                             batch_size=args.bs,
                                             drop_last=False,
                                             num_workers=dataloader_num_workers,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             pin_memory=True,)

    return dataloader
