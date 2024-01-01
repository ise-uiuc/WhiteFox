
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, num_qk, num_v, dropout_p):
        super().__init__()
        self.qkv = torch.nn.Linear(dim, num_heads * (num_qk + num_v))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, dropout_p):
        qkv = self.qkv(query)
        qkv = torch.chunk(qkv, chunks=2, dim=-1)
        q, k, v = map(lambda t: torch.transpose(t, 1, 2), qkv)
        scale_factor = (dim ** -0.5)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        output = torch.transpose(output, 1, 2)
        return output 

# Initializing the model
m = Model(dim=64, num_heads=2, num_qk=8, num_v=16, dropout_p=0.1)

# Inputs to the model
query = torch.randn(1, 8, 64)
key = torch.randn(1, 16, 64)
value = torch.randn(1, 16, 64)
dropout_p = 0.1
