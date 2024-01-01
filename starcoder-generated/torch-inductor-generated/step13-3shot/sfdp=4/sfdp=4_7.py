
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        size = 64
        self.query = torch.nn.Linear(size, size)
        self.key = torch.nn.Linear(size, size)
        self.value = torch.nn.Linear(size, size)
 
    def forward(self, query, key, value, mask):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        v_shape = v.shape
        v = v.reshape([-1, v_shape[1], v_shape[2]])
        size = q.shape
        k = k.reshape([size[0], size[1], size[2], size[3]])
        attn_mask = mask.reshape(*size, 1)
        qk = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(size[-1]))
        return torch.matmul(torch.reshape(torch.softmax(qk + attn_mask, dim=-2), v_shape), v)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(12, 64, 8)
x2 = torch.randn(12, 64, 8)
x3 = torch.randn(12, 64, 8)
mask = torch.triu(torch.ones(16, 31, 31), diagonal=0)  # upper triangular matrix
