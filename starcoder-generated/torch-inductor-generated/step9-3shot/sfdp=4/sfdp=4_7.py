
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout, bias=True):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.attn_func = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)
 
    def forward(self, query, key, value, mask):
        attention_output, attention_weight = self.attn_func(query, key, value, mask, need_weights=True)
        output = self.dropout(attention_output)
        return output, attention_weight

# Initializing the model
m = Model(d_model=128, nhead=4, dropout=0.2)

# Inputs to the model
query = torch.rand(16, 1, 128)
key = torch.rand(16, 20, 128)
value = torch.rand(16, 20, 128)
mask = torch.randint(low=0, high=2, size=(16, 1, 1, 20))
__output__, __output2__ = m(query, key, value, mask)

