
class Model(torch.nn.Module):
    def __init__(self, num_heads, query_len, hidden_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.query = torch.nn.Parameter(torch.zeros(num_heads, query_len, hidden_size))
        self.key = torch.nn.Parameter(torch.zeros(num_heads, query_len, hidden_size))
        self.value = torch.nn.Parameter(torch.zeros(num_heads, query_len, hidden_size))
        self.dropout = dropout
 
    def forward(self, query, key, value, batch_size):
        x1 = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt(torch.tensor(float(self.head_size), dtype=torch.float32))
        v1 = x1.div(inv_scale_factor)
        v2 = torch.nn.functional.softmax(v1, dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=self.dropout)
        return torch.matmul(v3, value).reshape(batch_size, -1, v3.shape[-1])

# Initializing the model
m = Model(4, 16, 128, 0.5)

# Inputs to the model
query = torch.randn(64, 16, 128)
key = torch.randn(64, 16, 128)
value = torch.randn(64, 16, 128)
batch_size = 64
m(query, key, value, batch_size)[:,:,0]*0.5

