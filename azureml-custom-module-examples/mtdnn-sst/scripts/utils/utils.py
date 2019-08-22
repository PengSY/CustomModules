# Copyright (c) Microsoft. All rights reserved.
import random
import torch
import numpy
import subprocess
import logging
from .metrics import calc_metrics


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_environment(seed, set_cuda=False):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and set_cuda:
        torch.cuda.manual_seed_all(seed)


def patch_var(v, cuda=True):
    if cuda:
        v = v.cuda(non_blocking=True)
    return v


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_pip_env():
    result = subprocess.call(["pip", "freeze"])
    return result


def eval_model(model, data, metric_meta, use_cuda=True, with_label=True):
    data.reset()
    if use_cuda:
        model.cuda()
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for batch_meta, batch_data in data:
        score, pred, gold = model.predict(batch_meta, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_meta['uids'])
    if with_label:
        metrics = calc_metrics(metric_meta, golds, predictions, scores)
    return metrics, predictions, scores, golds, ids


def setup_logger(logger_name='logger', log_file=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s] - %(message)s')
    if log_file:
        fh = logging.FileHandler(log_file, 'w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class MTDNNSSTConstants:
    TextColumn = "text"
    LabelColumn = "label"
    UidColumn = "uid"
    TokenColumn = "token_id"
    TypeIdColumn = "type_id"
    ScoreColumn = 'Predict label'
    IdColumn = 'Id'

    PreprocessedFile = "preprocessed_data.parquet"
    InputFile = "data.dataset.parquet"
    ScoreFile = 'scored_file.parquet'
    TrainedModel = 'trained_model.pt'
    ModelMetaFile = "model_meta.json"
    ConfigFile = 'config.json'
    InitCheckpointFile = "mt_dnn_large_uncased.pt"

    SSTMetric = ('ACC',)
