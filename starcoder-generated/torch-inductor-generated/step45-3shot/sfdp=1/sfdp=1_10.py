
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.wq = torch.nn.Linear(dim, num_heads * dim)
        self.wk = torch.nn.Linear(dim, num_heads * dim)
        self.wv = torch.nn.Linear(dim, num_heads * dim)
        self.wo = torch.nn.Linear(dim, num_heads * dim)
        self.dropout = dropout
 
    def forward(self, query, key, value, mask):
        dim_key = key.size(-1)
        v_qk = torch.matmul(query, key.transpose(-2, -1))
        v_qk = v_qk.div(dim_key ** 0.5)
        v_dropout = torch.nn.functional.dropout(v_softmax, p=self.dropout)
        v_output = self.wo(v_dropout)
        return v_output

# Initializing the model
m = Model(dim, num_heads, ffn_dim, dropout)

# Inputs to the model
query = torch.randn(1, num_heads, dim)
key = torch.randn(1, num_heads, dim)
value = torch.randn(1, num_heads, dim)
mask = torch.empty(1, 1, 1024, dtype=torch.float32).bernoulli_(0.5)
