
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(64, 16)
        self.m2 = torch.nn.Linear(16, 32)
        self.m3 = torch.nn.Linear(32, 16)
        self.m4 = torch.nn.Linear(16, 8)
        self.m5 = torch.nn.Linear(8, 4)
        self.m6 = torch.nn.Linear(4, 2)
        self.m7 = torch.nn.Linear(2, 16)
        self.m8 = torch.nn.Linear(16, 32)
        self.m9 = torch.nn.Linear(32, 64)
 
    def forward(self, x1):
        v1 = self.m1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.m2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.m3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.m4(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.m5(v8)
        v10 = torch.sigmoid(v9)
        v11 = self.m6(v10)
        v12 = torch.sigmoid(v11)
        v13 = self.m7(v12)
        v14 = torch.sigmoid(v13)
        v15 = v14.transpose(-2, -1)
        v16 = self.m8(v14)
        v17 = torch.sigmoid(v16)
        v18 = self.m9(v17)
        v19 = torch.sigmoid(v18)
        v20 = v19.transpose(-2, -1)
        v21 = torch.matmul(v20, v15) / np.sqrt(16)
        v22 = F.softmax(v21, dim=-1)
        v23 = torch.matmul(v22, v20)
        return v23

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 64)
