
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_qkv = torch.nn.Linear(64, 512)
        self.scale = math.sqrt(512)
    
    def forward(x64):
        q, k, v = self.dense_qkv(x64).chunk(3, dim=-1)
        q /= self.scale
        k /= self.scale
        out = torch.matmul(q, k.transpose(-2, -1))
        return out

# Initializing the model
m = Model()

# Inputs to the model
x64 = torch.randn(5, 64)
