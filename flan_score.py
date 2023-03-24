# %%
import torch
import torch.nn as nn
import traceback
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

class FLANScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='google/flan-t5-base'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,torch_dtype=torch.float16)
        self.model.eval()
        self.model.to(device)
        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self):
        """ Load model from paraphrase finetuning """
        self.model.load_state_dict(torch.load('models/bart.pth', map_location=self.device))

    def inverse_frequency(self,matrix):
        # Compute frequency of each item in each row
        freq_matrix = torch.zeros((matrix.shape[0], matrix.max()+1), dtype=torch.float)
        for i in range(matrix.shape[0]):
            row = matrix[i]
            unique_values, counts = torch.unique(row, return_counts=True)
            freq_matrix[i, unique_values] = counts.float()
        freq_matrix[freq_matrix == 0] = 1  # avoid division by zero

        # Compute inverse frequency of each item in each row
        inv_freq_matrix = 1 / freq_matrix

        # Create new matrix to store inverse frequencies
        result_matrix = inv_freq_matrix.gather(1, matrix)
        return result_matrix
    def score(self, srcs, tgts, batch_size,weighted=False):
        """ Score a batch of examples """
        score_list = []
        for i in tqdm(range(0, len(srcs), batch_size)):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
     
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)
                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)

                    if weighted:
                        inverse_frequency=self.inverse_frequency(tgt_tokens)
                        loss = torch.mean(loss*inverse_frequency.to(self.device),dim=1)
                        curr_score_list = [torch.exp(-x).item() for x in loss]    
                    else:
                        loss = loss.sum(dim=1) / tgt_len
                        curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list
