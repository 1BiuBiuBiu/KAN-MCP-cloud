import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.nn import MSELoss

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW
from KANMCP import KANMCP

import global_configs
from global_configs import DEVICE
import experiment_utils as exp_utils

import warnings
from min_norm_solvers import MinNormSolver
import torch.nn.functional as F
import re

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="google-bert")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei", "simsv2"], default="simsv2")
parser.add_argument("--max_seq_length", type=int, default=768)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--text_learning_rate", type=float, default=1e-3)
parser.add_argument("--other_learning_rate", type=float, default=1e-3)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=1025)
parser.add_argument("--m_dim", type=int, default=1024)

parser.add_argument('--kan_hidden_neurons', type=int, default=3)
parser.add_argument('--compressed_dim', type=int, default=3)

parser.add_argument('--weight1', type=float, default=1)
parser.add_argument('--weight2', type=float, default=1)

parser.add_argument('--gamma', type=float, default=1.5)
parser.add_argument('--tqdm_disable', type=bool, default=False)
parser.add_argument('--use_MMPareto', type=bool, default=False)

parser.add_argument('--use_DRDMIB_or_AE', type=int, default=0)
parser.add_argument('--use_KAN_or_MLP', type=bool, default=True)
parser.add_argument('--run_base_dir', type=str, default="/root/autodl-tmp/runs")

args = parser.parse_args()

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM,global_configs.TEXT_DIM)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            visual,
            acoustic,
            input_mask,
            segment_ids,
            label_id,
            text_label,
            audio_label,
            visual_label,
    ):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.text_label = text_label
        self.audio_label = audio_label
        self.visual_label = visual_label



def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)


        prepare_input = prepare_deberta_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )


        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
                text_label=label_id,
                audio_label=label_id,
                visual_label=label_id,
           )
        )
    return features


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)


def get_appropriate_dataset(data):
    # simsv2 is stored as a dict of numpy arrays (keys like text_bert/audio/vision)
    # while mosi/mosei are lists of tuples. Branch to handle both structures.
    if isinstance(data, dict):
        labels = data["regression_labels"]
        text_labels = data.get("text_labels", labels)
        vision_labels = data.get("vision_labels", labels)
        audio_labels = data.get("audio_labels", labels)
        text_bert = data["text"]
        audio = data["audio"]
        vision = data["vision"]
        audio_lens = data["audio_lengths"]
        vision_lens = data["vision_lengths"]

        features = []
        target_len = args.max_seq_length

        for idx in range(len(labels)):
            bert_block = text_bert[idx]
            input_ids = bert_block[0].astype(np.int64).tolist()
            input_mask = bert_block[1].astype(np.int64).tolist()
            segment_ids = bert_block[2].astype(np.int64).tolist()

            # Pad/truncate text to target length
            def _fix_length(seq, pad_val=0):
                if len(seq) >= target_len:
                    return seq[:target_len]
                return seq + [pad_val] * (target_len - len(seq))

            input_ids = _fix_length(input_ids)
            input_mask = _fix_length(input_mask)
            segment_ids = _fix_length(segment_ids)

            # Resample acoustic / visual sequences to the same length as text
            a_valid = int(audio_lens[idx])
            v_valid = int(vision_lens[idx])

            acoustic_seq = audio[idx][: max(a_valid, 1)]
            visual_seq = vision[idx][: max(v_valid, 1)]

            a_idx = np.linspace(0, max(a_valid - 1, 0), num=target_len).astype(int)
            v_idx = np.linspace(0, max(v_valid - 1, 0), num=target_len).astype(int)

            acoustic_fixed = np.asarray(acoustic_seq[a_idx], dtype=np.float32)
            visual_fixed = np.asarray(visual_seq[v_idx], dtype=np.float32)

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    visual=visual_fixed,
                    acoustic=acoustic_fixed,
                    label_id=float(labels[idx]),
                    text_label=float(text_labels[idx]),
                    audio_label=float(audio_labels[idx]),
                    visual_label=float(vision_labels[idx]),
                )
            )
    else:
        tokenizer = get_tokenizer(args.model)
        features = convert_to_features(data, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)
    all_text_label_ids = torch.tensor(np.array([f.text_label for f in features]), dtype=torch.float)
    all_audio_label_ids = torch.tensor(np.array([f.audio_label for f in features]), dtype=torch.float)
    all_visual_label_ids = torch.tensor(np.array([f.visual_label for f in features]), dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_label_ids,
        all_text_label_ids,
        all_audio_label_ids,
        all_visual_label_ids,
    )
    return dataset



def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"] if "dev" in data else data["valid"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
            int(
                len(train_dataset) / args.train_batch_size /
                args.gradient_accumulation_step
            )
            * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=False,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # print("Seed: {}".format(seed))


def prep_for_training(num_train_optimization_steps: int):
    model = KANMCP.from_pretrained(args.model, multimodal_config=args, num_labels=1)

    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    text_parameters = []
    other_parameters = []

    for item in param_optimizer:
        if "Deberta" in item[0].split("."):
            text_parameters.append(item[0])
        elif "TEncoder" in item[0].split("."):
            text_parameters.append(item[0])
        else:
            other_parameters.append(item[0])

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in text_parameters)
            ],
            "weight_decay": 0.01,
            "lr": args.text_learning_rate,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in other_parameters)
            ],
            "weight_decay": 0.01,
            "lr": args.other_learning_rate,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    tr_audio_loss = 0
    tr_visual_loss = 0
    tr_text_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    record_names_audio = []
    record_names_visual = []
    record_names_text = []
    for name, param in model.named_parameters():
        if 'AEncoder' in name:
            if 'decoder' in name:
                continue
            record_names_audio.append((name, param))
            continue
        if 'VEncoder' in name:
            if 'decoder' in name:
                continue
            record_names_visual.append((name, param))
            continue
        if 'TEncoder' in name:
            if 'decoder' in name:
                continue
            record_names_text.append((name, param))
            continue
        if 'Deberta' in name:
            record_names_text.append((name, param))
            continue

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.tqdm_disable)):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids, text_label_ids, audio_label_ids, visual_label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

        outputs = model(
            input_ids,
            visual_norm,
            acoustic_norm,
            label_ids,
            text_label_ids,
            audio_label_ids,
            visual_label_ids,
        )

        logits = outputs["logits"]
        loss_t = outputs["loss_t"]
        loss_a = outputs["loss_a"]
        loss_v = outputs["loss_v"]

        loss_fct = MSELoss()

        if args.use_MMPareto:
            loss_mm = loss_fct(logits.view(-1), label_ids.view(-1))

            losses = [loss_mm, loss_t, loss_v, loss_a]
            all_loss = ['both', 'text', 'visual', 'audio']

            grads_text = {}
            grads_audio = {}
            grads_visual = {}

            for idx, loss_type in enumerate(all_loss):
                loss = losses[idx]
                loss.backward(retain_graph=True)

                if loss_type == 'visual':
                    for tensor_name, param in record_names_visual:
                        if loss_type not in grads_visual.keys():
                            grads_visual[loss_type] = {}
                        grad_tensor = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
                        grads_visual[loss_type][tensor_name] = grad_tensor
                    grads_visual[loss_type]["concat"] = torch.cat(
                        [grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])
                elif loss_type == 'audio':
                    for tensor_name, param in record_names_audio:
                        if loss_type not in grads_audio.keys():
                            grads_audio[loss_type] = {}
                        grad_tensor = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
                        grads_audio[loss_type][tensor_name] = grad_tensor
                    grads_audio[loss_type]["concat"] = torch.cat(
                        [grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])
                elif loss_type == 'text':
                    for tensor_name, param in record_names_text:
                        if loss_type not in grads_text.keys():
                            grads_text[loss_type] = {}
                        grad_tensor = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
                        grads_text[loss_type][tensor_name] = grad_tensor
                    grads_text[loss_type]["concat"] = torch.cat(
                        [grads_text[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_text])

                else:
                    for tensor_name, param in record_names_text:
                        if loss_type not in grads_text.keys():
                            grads_text[loss_type] = {}
                        grad_tensor = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
                        grads_text[loss_type][tensor_name] = grad_tensor
                    grads_text[loss_type]["concat"] = torch.cat(
                        [grads_text[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_text])

                    for tensor_name, param in record_names_audio:
                        if loss_type not in grads_audio.keys():
                            grads_audio[loss_type] = {}
                        grad_tensor = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
                        grads_audio[loss_type][tensor_name] = grad_tensor
                    grads_audio[loss_type]["concat"] = torch.cat(
                        [grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])

                    for tensor_name, param in record_names_visual:
                        if loss_type not in grads_visual.keys():
                            grads_visual[loss_type] = {}
                        grad_tensor = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
                        grads_visual[loss_type][tensor_name] = grad_tensor
                    grads_visual[loss_type]["concat"] = torch.cat(
                        [grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])

                optimizer.zero_grad()

            this_cos_text = F.cosine_similarity(grads_text['both']["concat"], grads_text['text']["concat"], dim=0)
            this_cos_audio = F.cosine_similarity(grads_audio['both']["concat"], grads_audio['audio']["concat"], dim=0)
            this_cos_visual = F.cosine_similarity(grads_visual['both']["concat"], grads_visual['visual']["concat"],
                                                  dim=0)

            text_task = ['both', 'text']
            audio_task = ['both', 'audio']
            visual_task = ['both', 'visual']

            # audio_k[0]: weight of multimodal loss
            # audio_k[1]: weight of audio loss
            # if cos angle <0 , solve pareto
            # else use equal weight
            text_k = [0, 0]
            audio_k = [0, 0]
            visual_k = [0, 0]

            if this_cos_text > 0:
                text_k[0] = 0.5
                text_k[1] = 0.5
            else:
                text_k, min_norm = MinNormSolver.find_min_norm_element(
                    [list(grads_text[t].values()) for t in text_task])
                loss_t = args.weight2 * loss_t
            if this_cos_audio > 0:
                audio_k[0] = 0.5
                audio_k[1] = 0.5
            else:
                audio_k, min_norm = MinNormSolver.find_min_norm_element(
                    [list(grads_audio[t].values()) for t in audio_task])
                loss_a = args.weight2 * loss_a
            if this_cos_visual > 0:
                visual_k[0] = 0.5
                visual_k[1] = 0.5
            else:
                visual_k, min_norm = MinNormSolver.find_min_norm_element(
                    [list(grads_visual[t].values()) for t in visual_task])
                loss_v = args.weight2 * loss_v

            # 加上其他压缩损失，并给予较小的权重
            loss = loss_mm + args.weight1 * (loss_t + loss_v + loss_a)
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            loss.backward()

            gamma = args.gamma
            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer = re.split('[_.]', str(name))
                    # if('head' in layer):
                    #     continue
                    if 'TEncoder' in layer and 'decoder' not in layer:
                        three_norm = torch.norm(param.grad.data.clone())
                        new_grad = 2 * text_k[0] * grads_text['both'][name] + 2 * text_k[1] * grads_text['text'][name]
                        new_norm = torch.norm(new_grad)
                        diff = three_norm / new_norm
                        if diff > 1:
                            param.grad = diff * new_grad * gamma
                        else:
                            param.grad = new_grad * gamma

                    if 'Deberta' in layer:
                        three_norm = torch.norm(param.grad.data.clone())
                        new_grad = 2 * text_k[0] * grads_text['both'][name] + 2 * text_k[1] * grads_text['text'][name]
                        new_norm = torch.norm(new_grad)
                        diff = three_norm / new_norm
                        if diff > 1:
                            param.grad = diff * new_grad * gamma
                        else:
                            param.grad = new_grad * gamma

                    if 'AEncoder' in layer and 'decoder' not in layer:
                        three_norm = torch.norm(param.grad.data.clone())
                        new_grad = 2 * audio_k[0] * grads_audio['both'][name] + 2 * audio_k[1] * grads_audio['audio'][
                            name]
                        new_norm = torch.norm(new_grad)
                        diff = three_norm / new_norm
                        if diff > 1:
                            param.grad = diff * new_grad * gamma
                        else:
                            param.grad = new_grad * gamma

                    if 'VEncoder' in layer and 'decoder' not in layer:
                        three_norm = torch.norm(param.grad.data.clone())
                        new_grad = 2 * visual_k[0] * grads_visual['both'][name] + 2 * visual_k[1] * \
                                   grads_visual['visual'][name]
                        new_norm = torch.norm(new_grad)
                        diff = three_norm / new_norm
                        if diff > 1:
                            param.grad = diff * new_grad * gamma
                        else:
                            param.grad = new_grad * gamma
        else:
            loss = loss_fct(logits.view(-1), label_ids.view(-1)) + 0.1*(loss_t + loss_v + loss_a)
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            loss.backward()



        tr_loss += loss.item()
        tr_visual_loss += loss_v.item()
        tr_audio_loss += loss_a.item()
        tr_text_loss += loss_t.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return {
        "total": tr_loss / nb_tr_steps,
        "audio": tr_audio_loss / nb_tr_steps,
        "visual": tr_visual_loss / nb_tr_steps,
        "text": tr_text_loss / nb_tr_steps,
    }


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0

    # 11.30
    preds = []
    labels = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration", disable=True)):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids, text_label_ids, audio_label_ids, visual_label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            outputs = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                label_ids,
                text_label_ids,
                audio_label_ids,
                visual_label_ids,
            )

            logits = outputs["logits"]

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

            # 11.30
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, disable=True):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, label_ids, text_label_ids, audio_label_ids, visual_label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            outputs = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                label_ids,
                text_label_ids,
                audio_label_ids,
                visual_label_ids,
            )
            
            logits = outputs["logits"]

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero: bool = False):
    preds, y_test = test_epoch(model, test_dataloader)

    allowed_vals = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)

    def _nearest_allowed(values: np.ndarray) -> np.ndarray:
        """Snap continuous predictions to the nearest allowed label."""
        diff = np.abs(values[:, None] - allowed_vals[None, :])
        nearest_idx = np.argmin(diff, axis=1)
        return allowed_vals[nearest_idx]

    def _map_to_5_class(values: np.ndarray) -> np.ndarray:
        buckets = {
            -1.0: 0,
            -0.8: 0,
            -0.6: 1,
            -0.4: 1,
            -0.2: 1,
            0.0: 2,
            0.2: 3,
            0.4: 3,
            0.6: 3,
            0.8: 4,
            1.0: 4,
        }

        def _bucket(v: float) -> int:
            # guard floating drift by rounding to one decimal and falling back to nearest allowed value
            v_rounded = float(np.round(v, 1))
            if v_rounded in buckets:
                return buckets[v_rounded]
            # fallback: snap to nearest allowed then bucket
            nearest = float(_nearest_allowed(np.asarray([v], dtype=np.float32))[0])
            return buckets[nearest]

        return np.array([_bucket(float(v)) for v in values], dtype=np.int64)

    def _map_to_3_class(values: np.ndarray) -> np.ndarray:
        return np.array([
            0 if v < 0 else 1 if v == 0 else 2
            for v in values
        ], dtype=np.int64)

    def _map_to_2_class(values: np.ndarray) -> np.ndarray:
        return np.array([0 if v < 0 else 1 for v in values], dtype=np.int64)

    snapped_preds = _nearest_allowed(np.asarray(preds, dtype=np.float32))
    snapped_labels = _nearest_allowed(np.asarray(y_test, dtype=np.float32))

    non_zeros = np.array([i for i, e in enumerate(snapped_labels) if e != 0 or use_zero])

    mae = np.mean(np.absolute(snapped_preds - snapped_labels))
    corr = np.corrcoef(snapped_preds, snapped_labels)[0][1]

    # 5-class metrics
    preds_5 = _map_to_5_class(snapped_preds)
    labels_5 = _map_to_5_class(snapped_labels)
    acc5 = accuracy_score(labels_5, preds_5)

    # 3-class metrics
    preds_3 = _map_to_3_class(snapped_preds)
    labels_3 = _map_to_3_class(snapped_labels)
    acc3 = accuracy_score(labels_3, preds_3)

    # 2-class metrics (negative vs non-negative). Neutral joins non-negative.
    preds_2 = _map_to_2_class(snapped_preds[non_zeros])
    labels_2 = _map_to_2_class(snapped_labels[non_zeros])
    acc2 = accuracy_score(labels_2, preds_2)
    f1_2 = f1_score(labels_2, preds_2, average="weighted")

    return {
        "acc5": acc5,
        "acc3": acc3,
        "acc2": acc2,
        "f1": f1_2,
        "mae": float(mae),
        "corr": float(corr),
    }


def train(
        model,
        train_dataloader,
        validation_dataloader,
        test_data_loader,
        optimizer,
        scheduler,
        run_meta=None,
        swan_run=None,
):
    min_eval_loss = 1000
    test_result = {
        "acc5": 0.0,
        "acc3": 0.0,
        "acc2": 0.0,
        "f1": 0.0,
        "mae": 0.0,
        "corr": 0.0,
        "epoch": 0,
    }
    metrics_history = []
    best_model_path = None

    for epoch_i in range(int(args.n_epochs)):
        train_losses = train_epoch(model, train_dataloader, optimizer, scheduler)
        eval_loss = eval_epoch(model, validation_dataloader)
        metrics = test_score_model(model, test_data_loader)

        # 输出
        print("TRAIN: epoch:{}, train_loss:{}, eval_loss:{}".format(epoch_i + 1, train_losses["total"], eval_loss))
        print(
            "TEST: acc5: {}, acc3: {}, acc2: {}, f1: {}, mae: {}, corr: {}".format(
                metrics["acc5"], metrics["acc3"], metrics["acc2"], metrics["f1"], metrics["mae"], metrics["corr"]
            )
        )

        epoch_metrics = {
            "epoch": epoch_i + 1,
            "train_loss": float(train_losses["total"]),
            "eval_loss": float(eval_loss),
            "train_loss_audio": float(train_losses["audio"]),
            "train_loss_visual": float(train_losses["visual"]),
            "train_loss_text": float(train_losses["text"]),
            "acc5": float(metrics["acc5"]),
            "acc3": float(metrics["acc3"]),
            "acc2": float(metrics["acc2"]),
            "f1": float(metrics["f1"]),
            "mae": float(metrics["mae"]),
            "corr": float(metrics["corr"]),
        }
        metrics_history.append(epoch_metrics)
        if run_meta:
            metrics_dir = os.path.join(run_meta["artifacts"], "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            exp_utils.save_json(epoch_metrics, os.path.join(metrics_dir, f"test_epoch_{epoch_i + 1}.json"))
        exp_utils.log_swanlab_metrics(
            swan_run,
            {
                "loss/train": epoch_metrics["train_loss"],
                "loss/train_audio": epoch_metrics["train_loss_audio"],
                "loss/train_visual": epoch_metrics["train_loss_visual"],
                "loss/train_text": epoch_metrics["train_loss_text"],
                "loss/eval": epoch_metrics["eval_loss"],
                "metrics/acc5": epoch_metrics["acc5"],
                "metrics/acc3": epoch_metrics["acc3"],
                "metrics/acc2": epoch_metrics["acc2"],
                "metrics/f1": epoch_metrics["f1"],
                "metrics/mae": epoch_metrics["mae"],
                "metrics/corr": epoch_metrics["corr"],
            },
            step=epoch_i + 1,
        )

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            test_result["acc5"] = metrics["acc5"]
            test_result["acc3"] = metrics["acc3"]
            test_result["acc2"] = metrics["acc2"]
            test_result["f1"] = metrics["f1"]
            test_result["mae"] = metrics["mae"]
            test_result["corr"] = metrics["corr"]
            test_result["epoch"] = epoch_i + 1
            if run_meta:
                best_model_path = os.path.join(run_meta["artifacts"], "best_model.pt")
                exp_utils.save_model_checkpoint(model, best_model_path)
                exp_utils.save_json(test_result, os.path.join(run_meta["run_dir"], "best_result.json"))

        if epoch_i + 1 == args.n_epochs:
            print("====RESULT====")
            print(
                "acc5:{}, acc3:{}, acc2:{}, f1:{}, mae:{}, corr:{}".format(
                    test_result["acc5"],
                    test_result["acc3"],
                    test_result["acc2"],
                    test_result["f1"],
                    test_result["mae"],
                    test_result["corr"],
                )
            )

    return test_result, metrics_history, best_model_path


def main():
    warnings.filterwarnings('ignore', category=UserWarning)
    print(args)

    set_random_seed(args.seed)

    run_meta = exp_utils.create_run_dirs(args.dataset, base_dir=args.run_base_dir)
    exp_utils.save_json(vars(args), os.path.join(run_meta["run_dir"], "args.json"))
    swan_run = exp_utils.init_swanlab_run(args, run_meta)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)
    print(model)
    test_result, metrics_history, best_model_path = train(
       model,
       train_data_loader,
       dev_data_loader,
       test_data_loader,
       optimizer,
       scheduler,
       run_meta=run_meta,
       swan_run=swan_run,
    )

    exp_utils.save_json(metrics_history, os.path.join(run_meta["run_dir"], "metrics_history.json"))
    last_model_path = os.path.join(run_meta["artifacts"], "last_model.pt")
    exp_utils.save_model_checkpoint(model, last_model_path)

    if args.use_KAN_or_MLP:
        kan_batch = next(iter(test_data_loader))
        contribution = exp_utils.plot_kan_contribution(
            model,
            kan_batch,
            args.compressed_dim,
            DEVICE,
            os.path.join(run_meta["plots"], "kan_contribution.png"),
        )
        exp_utils.save_json(contribution, os.path.join(run_meta["run_dir"], "kan_contribution.json"))
        exp_utils.log_swanlab_metrics(
            swan_run,
            {f"kan/{k}": v for k, v in contribution["per_modality"].items()},
            step=int(args.n_epochs) + 1,
        )
        exp_utils.plot_kan_tree(
            model,
            kan_batch,
            DEVICE,
            os.path.join(run_meta["plots"], "kan_tree.png"),
        )
        exp_utils.plot_kan_model_diagram(
            model,
            kan_batch,
            DEVICE,
            os.path.join(run_meta["plots"], "kan_model.png"),
        )

    if swan_run is not None and exp_utils.swanlab is not None:
        exp_utils.log_swanlab_metrics(
            swan_run,
            {f"best/{k}": float(v) for k, v in test_result.items() if k != "epoch"},
            step=int(args.n_epochs) + 1,
        )
        exp_utils.swanlab.finish()

    return model


if __name__ == '__main__':
    main()

