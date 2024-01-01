
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1280, 2560)
        self.linear2 = torch.nn.Linear(2560, 2560)
        self.linear3 = torch.nn.Linear(2560, 2560)
        self.linear4 = torch.nn.Linear(2560, 1280)

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 * torch.clamp(v1 + 3, 0, 6) / 6
        v3 = self.linear2(v2)
        v4 = v3 * torch.clamp(v3 + 3, 0, 6) / 6
        v5 = self.linear3(v4)
        v6 = v5 * torch.clamp(v5 + 3, 0, 6) / 6
        v7 = self.linear4(v6)
        v8 = v7 * torch.clamp(v7 + 3, 0, 6) / 6 
        return v8
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = x1 = torch.randn(1, 1280)
