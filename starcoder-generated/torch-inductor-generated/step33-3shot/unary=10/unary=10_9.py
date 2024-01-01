
class Linear0Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 3, bias=False)
        self.relu = torch.nn.ReLU6()
 
    def forward(self, x2):
        v7 = self.linear1(x2)
        v8 = v7 + 3
        v9 = self.relu(v8)
        return v9
class Linear1Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 3, bias=False)
        self.relu = torch.nn.ReLU6()
 
    def forward(self, x2):
        v7 = self.linear1(x2)
        v8 = v7 + 3
        v9 = self.relu(v8)
        return v9
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = Linear0Model()
        self.linear1 = Linear1Model()
 
    def forward(self, x2):
        l1 = self.linear0(x2)
        l2 = self.linear1(x2)
        return l1, l2

# Initializing the model
m = LinearModel()

# Inputs to the model
x2 = torch.randn(1, 10)
