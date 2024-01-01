
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, tgt, memory):
        tgt2 = self.multihead_attn(tgt, memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

# Initializing the model
m = Model(d_model = 512, nhead = 8, num_encoder_layers = 6, dim_feedforward = 512, dropout = 0.1)

# Inputs to the model
x1 = torch.randn(10, 32, 512)
x2 = torch.randn(20, 32, 512)
___output___ = m(x1, x2)

