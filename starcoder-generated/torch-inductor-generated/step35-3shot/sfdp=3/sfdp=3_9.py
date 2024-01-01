
class Model(torch.nn.Module):
    def __init__(self, num_heads, qkv_dim):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.scale_factor = (qkv_dim/ num_heads) ** (0.5) # Factor used to scale dot product
        self.query = torch.nn.Linear(qkv_dim, qkv_dim)
        self.key = torch.nn.Linear(qkv_dim, qkv_dim)
        self.value = torch.nn.Linear(qkv_dim, qkv_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2).transpose(-2, -1)
        v = self.value(x2)
        qk = torch.matmul(q, k)
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        v1 = dropout_qk.matmul(v)
        return v1

# Initializing the model
m = Model(num_heads, qkv_dim).to(device)

# Inputs to the model
x1 = torch.randn(batch_size, query_len, qkv_dim).to(device)
x2 = torch.randn(batch_size, key_len, qkv_dim).to(device)
