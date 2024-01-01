
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.out_dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key):
        scale_factor = 1. / math.sqrt(self.dim)
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(key)

        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)

        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)

        return self.out_dropout(dropout_qk.matmul(v))

# Initializing the model
m = Model(dim=128, num_heads=8, dropout=0.1)

# Inputs to the model
query = torch.randn(1, 8, 128)
key = torch.randn(1, 8, 128)
