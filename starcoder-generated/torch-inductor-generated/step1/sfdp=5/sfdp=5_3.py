
from torch import nn
class Model(torch.nn.Module):
    def __init__(self, batch_size = 1, hidden = 8, n_seq = 10, n_head = 2, hidden_per_head = 4, dropout = 0.5):
        super().__init__()

        self.multi_head_attntion = nn.MultiheadAttention(self.hparams.hidden, self.hparams.n_head, dropout=dropout)

    def forward(self, x, attn_mask = None):
      attn_output, _ = self.multi_head_attntion(x, x, x, attn_mask)
      return attn_output

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(1, 10, 8)
attn_mask = torch.randn(1, 10, 10)
