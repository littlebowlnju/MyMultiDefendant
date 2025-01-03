import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import jsonlines
from utils import get_charge_article_penalty_labels

class MultiDefendantJudgmentBaseDataset(Dataset):
    # used when validating and testing!!
    def __init__(self, data_path, tokenizer_name, max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._load_data(data_path)
    
    def _load_data(self, data_path):
        # load data from data_path
        self.data = []
        with jsonlines.open(data_path, "r") as reader:
            for case in reader:
                self.data.append(case)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        obj = self.data[index]
        defendant_input = f"被告人{obj["defendant"]}犯罪事实：{obj["extracted_fact"]}"
        inputs = self.tokenizer(defendant_input, padding="max_length", truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        charge_label, article_label, penalty_label = get_charge_article_penalty_labels(obj["judgment"])
        return {"input_ids": inputs["input_ids"].squeeze(0), \
                "attention_mask": inputs["attention_mask"].squeeze(0), \
                "charge_labels": torch.tensor(charge_label, dtype=torch.float16), \
                "article_labels": torch.tensor(article_label, dtype=torch.float16), \
                "penalty_labels": torch.tensor(penalty_label, dtype=torch.float16)}

# 预加载数据tokenize后保存？


class MultiDefendantJudgmentPairDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._load_data(data_path)
    
    def _load_data(self, data_path):
        # preload data from data_path
        self.data = []
        with jsonlines.open(data_path, "r") as reader:
            for pair in reader:
                self.data.append(pair)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # do tokenize and other stuff here
        pair_obj = self.data[index]
        defendant1_name, defendant2_name = pair_obj["defendant1"], pair_obj["defendant2"]
        defendant1_fact, defendant2_fact = pair_obj["defendant1_fact"], pair_obj["defendant2_fact"]
        defendant1_input = f"被告人{defendant1_name}犯罪事实：{defendant1_fact}"
        defendant2_input = f"被告人{defendant2_name}犯罪事实：{defendant2_fact}"
        defendant1_inputs = self.tokenizer(defendant1_input, padding="max_length", truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        defendant2_inputs = self.tokenizer(defendant2_input, padding="max_length", truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        # 合并两个，每个item返回[x, 2]大小的tensor
        defendant1_charge, defendant1_article, defendant1_penalty = get_charge_article_penalty_labels(pair_obj["defendant1_outcome"])
        defendant2_charge, defendant2_article, defendant2_penalty = get_charge_article_penalty_labels(pair_obj["defendant2_outcome"])
        return {"input_ids": torch.stack(defendant1_inputs["input_ids"].squeeze(0), defendant2_inputs["input_ids"].squeeze(0), dim=1), \
                "attention_mask": torch.stack(defendant1_inputs["attention_mask"].squeeze(0), defendant2_inputs["attention_mask"].squeeze(0), dim=1), \
                "charge_labels": torch.tensor([defendant1_charge, defendant2_charge], dtype=torch.float16), \
                "article_labels": torch.tensor([defendant1_article, defendant2_article], dtype=torch.float16), \
                "penalty_labels": torch.tensor([defendant1_penalty, defendant2_penalty], dtype=torch.float16),
                # return relation to see whether to add relative penalty loss
                "relation": int(pair_obj["relation"] == "lead")}
        