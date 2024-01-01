
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, bias=False)
        self.add = torch.tensor(3, dtype=torch.float32)
    
    def forward(self, x):
        v1 = x.reshape(1, 16)
        v2 = self.linear(v1)
        v3 = v2 + self.add
        v4 = v3.clamp_min(0)
        v5 = v4.clamp_max(6)
        v6 = v2 * v5
        v7 = v6.reshape(32)
        return v7
 
# Initializing the model
m = Model()
 
# Inputs to the model
x = torch.randn(1, 16)
