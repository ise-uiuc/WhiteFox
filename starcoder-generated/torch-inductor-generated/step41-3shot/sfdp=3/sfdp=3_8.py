
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.scale_factor = (dim // num_heads) ** -0.5
        self.dim = dim
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.qwk = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.vw = nn.Linear(dim, dim)
 
    def forward(self, query, key, value):
        b = query.shape[0]
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model(dim=32, num_heads=4, dropout=0.0)

# Inputs to the model
query = torch.randn(1, 4, 10, 32)
key = torch.randn(1, 4, 2, 32)
value = torch.randn(1, 4, 2, 32)
