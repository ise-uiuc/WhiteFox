
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(128, 288)
        self.k = torch.nn.Linear(13, 288)
        self.v = torch.nn.Linear(13, 288)
 
    def forward(self, q, k, v, mask):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        qk = q @ k.transpose(-2, -1) / math.sqrt(288)
        qk = qk + mask
        attn = torch.softmax(qk, dim=-1)
        output = attn @ v
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 128)
k = torch.randn(1, 13)
v = torch.randn(1, 13)
mask = torch.randn(1, 1, 13)
