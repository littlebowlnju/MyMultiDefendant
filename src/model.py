import torch
import torch.nn as nn
from transformers import AutoModel
import lightning as pl

class JointCrimeFeatureExtractor(nn.Module):
    """
    Given the fact description embedding, process the embedding to extract the joint crime information embeddings.
    """
    def __init__(self, seq_len, hidden_len):
        super(JointCrimeFeatureExtractor, self).__init__()
        self.seq_len = seq_len
        self.hidden_len = hidden_len
        # TODO more complated model to extract features
        # 最终输出维度与输入维度保持一致
        self.linear = nn.Linear(hidden_len, hidden_len)

    def forward(self, x):
        # TODO 
        pass


class MultiDefendantJudgmentPredictor(nn.Module):
    def __init__(self, encoder_name, charge_num, article_num, penalty_num, max_seq_len=512, hidden_len=768):
        super().__init__()
        # encode fact description
        # this fact encoder can be replaced by something else
        self.fact_encoder = AutoModel.from_pretrained(encoder_name)
        self.fact_encoder.train()
        self.joint_crime_feature_extractor = JointCrimeFeatureExtractor(max_seq_len, hidden_len)
        # Fuse the fact description embedding and joint crime information embedding
        self.fuse_feature_layer = nn.MultiheadAttention(hidden_len, head=2, dropout=0.5)
        self.charge_classifier = nn.Linear(hidden_len, charge_num)
        self.article_classifier = nn.Linear(hidden_len, article_num)
        self.penalty_classifier = nn.Linear(hidden_len, penalty_num)
    
    def forward(self, input_ids, attention_mask):
        fact_embedding = self.fact_encoder(input_ids, attention_mask)
        joint_crime_feature = self.joint_crime_feature_extractor(fact_embedding)
        fused_feature = self.fuse_feature_layer(fact_embedding, joint_crime_feature)
        charge_logits = self.charge_classifier(fused_feature)
        article_logits = self.article_classifier(fused_feature)
        penalty_logits = self.penalty_classifier(fused_feature)
        return charge_logits, article_logits, penalty_logits
        
class MultiDefendantLJPRelativeModule(pl.LightningModule):
    def __init__(self, encoder_name, charge_num, article_num, penalty_num, max_seq_len=512, hidden_len=768):
        super().__init__()
        self.model = MultiDefendantJudgmentPredictor(encoder_name, charge_num, article_num, penalty_num, max_seq_len, hidden_len)
    
    def training_step(self, batch, batch_idx):
        # each time inputs two samples to calculate 'relative penalty loss'
        # (batch_size, max_seq_len, 2)
        input_ids = batch["input_ids"]
        # (batch_size, max_seq_len, 2)
        attention_mask = batch["attention_mask"]
        # (batch_size, label_class_num, 2)
        charge_labels, article_labels, penalty_labels = batch["charge_labels"], batch["article_labels"], batch["penalty_labels"]
        # 每个sample中的defendant1
        defendant1_input_ids = input_ids[:, :, 0].squeeze(-1)
        defendant1_attention_mask = attention_mask[:, :, 0].squeeze(-1)
        charge_outputs, article_outputs, penalty_outputs = self.model(defendant1_input_ids, defendant1_attention_mask)
        # charge_loss = 


    def configure_optimizers(self, lr=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
