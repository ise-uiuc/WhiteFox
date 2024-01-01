
class Model(torch.nn.Module):
    def __init__(self, n_heads, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        self.proj = torch.nn.Linear(in_features, emb_size)       
    def forward(self, q, k, v):
        qk = torch.matmul(query, key.transpose(-2, -1))
        dim_divider = len(k.shape) - 2
        scale_factor = (self.emb_size // self.n_heads)**-0.5
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(value)
        return output

# Initializing the Model
m = Model(3, 10)

# Input to the model
n = 1
in_features = 8
query = torch.randn(n, in_features)
key = torch.randn(n, in_features)
value = torch.randn(n, in_features)
