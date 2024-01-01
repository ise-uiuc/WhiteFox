
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, q, k, v, mask):
        v1 = self.linear(q).transpose(-2, -1) / math.sqrt(q.size(-1))
        v2 = v1.matmul(k.transpose(-2, -1))
        v3 = v2 + mask
        v4 = torch.softmax(v3, dim=-1)
        return (v4.matmul(v).to(torch.float32), )
    
# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 5, 64)
k = torch.randn(1, 6, 64)
v = torch.randn(1, 6, 64)
mask = torch.ones([1, 5, 6]).eq(0)
