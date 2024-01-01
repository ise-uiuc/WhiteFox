
class Model(torch.nn.Module):
    def __init__(self, n_seq, hidden_size):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.qkv_proj = nn.Linear(hidden_size, hidden_size*3)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
 
    def forward(self, input):
        qkv = self.qkv_proj(self.norm(input))
        *_, out = qkv.split(hidden_size,dim=-1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 30, hidden_size)
