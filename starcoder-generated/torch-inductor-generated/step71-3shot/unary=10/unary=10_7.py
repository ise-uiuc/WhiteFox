
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 32)
        self.linear5 = torch.nn.Linear(32, 7)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.linear2(v5)
        v7 = v6 + 3
        v8 = torch.clamp_min(v7, 0)
        v9 = torch.clamp_max(v8, 6)
        v10 = v9 / 6
        v11 = self.linear3(v10)
        v12 = v11 + 3
        v13 = torch.clamp_min(v12, 0)
        v14 = torch.clamp_max(v13, 6)
        v15 = v14 / 6
        v16 = self.linear4(v15)
        v17 = v16 + 3
        v18 = torch.clamp_min(v17, 0)
        v19 = torch.clamp_max(v18, 6)
        v20 = v19 / 6
        v21 = self.linear5(v20)
        return v21

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 1, 1)
