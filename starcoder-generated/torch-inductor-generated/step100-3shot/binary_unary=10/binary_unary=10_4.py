
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(895, 100)
        self.linear2 = torch.nn.Linear(100, 50)
        self.linear3 = torch.nn.Linear(50, 20)
 

    def forward(self, x5, x6, x7, x8, x9, x10):
        v1 = self.linear1(x5)
        v2 = v1 + x6
        v3 = F.relu(v2)
 
        v4 = self.linear2(v3)
        v5 = v4 + x7
        v6 = F.relu(v5)
 
        v7 = self.linear3(v6)
        v8 = v7 + x8
        v9 = F.relu(v8)
 
        v10 = self.linear3(v9)
        v11 = v10 + x9
        v12 = F.relu(v11)
 
        v13 = self.linear3(v12)
        v14 = v13 + x10
        v15 = F.relu(v14)
 
        return v15

# Initializing the model
n = Model()

# Inputs to the model
x1 = torch.randn(1, 1280)
x2 = torch.randn(1, 100)
x3 = torch.randn(1, 1280)
x4 = torch.randn(1, 100)
x5 = torch.randn(1, 1280)
x6 = torch.randn(1, 100)
x7 = torch.randn(1, 1280)
x8 = torch.randn(1, 100)
x9 = torch.randn(1, 1280)
x10 = torch.randn(1, 100)
