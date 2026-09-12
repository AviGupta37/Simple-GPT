import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42) 

vocab_size=65
class BigramLanguageModle(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)
        return logits

m=BigramLanguageModle(vocab_size)
out=m(xb,yb)
print(out.shape)