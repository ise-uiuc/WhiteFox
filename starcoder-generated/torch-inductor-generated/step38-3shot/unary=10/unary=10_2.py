
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(16, 8)
        self.l2 = torch.nn.Linear(8, 7)
        self.l3 = torch.nn.Linear(7, 6)
        self.l4 = torch.nn.Linear(6, 5)
        self.l5 = torch.nn.Linear(5, 4)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.l2(v5)
        v7 = v6 + 5
        v8 = torch.clamp_min(v7, 0)
        v9 = torch.clamp_max(v8, 6)
        v10 = v9 / 6
 
        v11 = self.l3(v10)
        v12 = v11 + 9
        v13 = torch.clamp_min(v12, 0)
        v14 = torch.clamp_max(v13, 6)
        v15 = v14 / 6
        v16 = self.l4(v15)
        v17 = v16 + 7
        v18 = torch.clamp_min(v17, 0)
        v19 = torch.clamp_max(v18, 6)
        v20 = v19 / 6
 
        v21 = self.l5(v20)
        v22 = v21 + 6
        v23 = torch.clamp_min(v22, 0)
        v24 = torch.clamp_max(v23, 6)
        v25 = v24 / 6
        return v25

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
