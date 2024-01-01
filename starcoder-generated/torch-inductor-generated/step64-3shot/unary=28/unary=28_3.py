
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear1 = torch.nn.Linear(32, 32, bias=False)
        self.linear2 = torch.nn.Linear(32, 32, bias=False)
        self.linear3 = torch.nn.Linear(32, 32, bias=False)
        self.linear4 = torch.nn.Linear(32, 32, bias=False)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = torch.clamp_min(v1, 2)
        v3 = torch.clamp_max(v2, 4)
        v4 = self.linear2(v3)
        v5 = torch.clamp_min(v4, 50000)
        v6 = torch.clamp_max(v5, 5000)
        v7 = self.linear3(v6)
        v8 = torch.clamp_min(v7, 850)
        v9 = torch.clamp_max(v8, 860)
        v10 = self.linear4(v9)
        return v10

# Initializing the model
m = Model(min_value=2, max_value=4)

# Inputs to the model
x = torch.randn(1, 32)
