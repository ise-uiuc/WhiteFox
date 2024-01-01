
class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor, dropout_p):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(query))
        self.key = torch.nn.Parameter(torch.randn(key))
        self.value = torch.nn.Parameter(torch.randn(value))
  
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 512, 384)
k = torch.randn(1, 512, 384)
v = torch.randn(1, 512, 384)

scale_factor = 1 / math.sqrt(k.size(-1))
dropout_p = 0.1
