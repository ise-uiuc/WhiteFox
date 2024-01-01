
class Model(torch.nn.Module):
    def __init__(self, query, key, value, dropout_p):
        super().__init__()
        query_dim = key_dim = value.size(-1)
        assert query.size() == (1, query_dim)
        assert key.size() == (1, key_dim)
        assert value.size() == (1, key_dim)
        assert dropout_p == 0.0
        self.scale_factor = math.sqrt(query_dim)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, q, k):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        output = softmax_qk.matmul(v)
        return output

# Initializing the model
q = torch.randn(1, 10)
k = torch.randn(1, 10)
v = torch.randn(1, 10)
dropout_p = 0.0
m = Model(q, k, v, dropout_p)

# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 128, 128)
x3 = torch.randn(2, 3, 256, 256)
