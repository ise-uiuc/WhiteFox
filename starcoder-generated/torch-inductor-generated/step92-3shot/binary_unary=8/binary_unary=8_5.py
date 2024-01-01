
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(9, 9)
        self.fc2 = torch.nn.Linear(9, 9)
        self.fc3 = torch.nn.Linear(9, 9)
        self.fc4 = torch.nn.Linear(9, 9)
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(v1)
        v3 = self.fc3(v2)
        v4 = self.fc4(v3)
        v5 = self.fc1(v4)
        v6 = self.fc2(v4)
        v7 = self.fc3(v4)
        v8 = self.fc4(v4)
        v9 = self.fc1(v4)
        v10 = self.fc2(v4)
        v11 = self.fc3(v4)
        v12 = self.fc4(v4)
        v13 = self.fc1(v4)
        v14 = self.fc2(v4)
        v15 = self.fc3(v4)
        v16 = self.fc4(v4)
        v17 = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12 + v13 + v14 + v15 + v16
        v18 = torch.relu(v17)
        return v18
# Inputs to the model
x1 = torch.randn(1, 9)
