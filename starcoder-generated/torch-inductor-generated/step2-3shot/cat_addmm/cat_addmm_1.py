
class Model2(torch.nn.Module):
    def __init__(self, dim1):
        super().__init__()
        self.dim1 = dim1
 
        self.fc1 = torch.nn.Linear(10, dim1)
        self.fc2 = torch.nn.Linear(10, dim1)
        self.fc3 = torch.nn.Linear(dim1, 10)
 
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.bn2 = torch.nn.BatchNorm1d(10)
 
    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = self.bn2(x1)
        v3 = torch.cat([v1, v2], 1)
        v4 = self.fc1(v3)
        v5 = self.fc2(v3)
        v6 = torch.addmm(v4, v5, torch.eye(self.dim1))
        v7 = self.fc3(v6)
        return v7

# Initializing the model
m = Model2(10)

# Inputs to the model
x1 = torch.randn(1, 10)
