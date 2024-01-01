
class Model(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
 
        self.layer1 = torch.nn.Linear(100, 100)
        self.act = torch.nn.GELU()
        self.layer2 = torch.nn.Linear(100, 100)
        self.layer3 = torch.nn.Linear(100, 100)
 
    def forward(self, x1):
        v1 = self.layer1(x1)
        v2 = self.act(v1)
        v3 = self.layer2(v2)
        v4 = self.layer3(v3)
        v5 = torch.matmul(v3, v4.transpose(-2, -1))
        return v5

# Initializing the model
m = Model(out_dim=100)

# Inputs to the model
x1 = torch.randn(1, 100)
