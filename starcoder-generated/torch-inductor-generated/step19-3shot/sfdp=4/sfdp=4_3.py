
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(64, 1024)
 
    def forward(self, x):
        qkv = self.qkv(x).reshape(1, 4, 32, -1)
        q, k, v = qkv[0:1, 0:1], qkv[0:1, 1:2], qkv[0:1, 2:3]
        q = q * (1.0 / math.sqrt(q.size(-1)))
        attn_mask = torch.tensor([[[-10000.0, 0.0, 0.0, 0.0],
              [0.0, -10000.0, 0.0, 0.0],
              [0.0, 0.0, -10000.0, 0.0],
              [0.0, 0.0, 0.0, -10000.0]]])
        attn_weight = torch.nn.functional.softmax(torch.matmul(q, torch.transpose(k, -2, -1)) + attn_mask, dim=-1)
        output = torch.matmul(attn_weight, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(4, 3, 64)
