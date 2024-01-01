
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2304, 1152)
        self.fc2 = torch.nn.Linear(1152, 1152)
        self.fc3 = torch.nn.Linear(1152, 1152)
        self.fc4 = torch.nn.Linear(1152, 1152)
        self.fc5 = torch.nn.Linear(1152, 1152)
        self.fc6 = torch.nn.Linear(1152, 1152)
        self.fc7 = torch.nn.Linear(1152, 1152)
        self.fc8 = torch.nn.Linear(1152, 1152)
        self.fc9 = torch.nn.Linear(1152, 1152)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(v1)
        v3 = self.fc3(v2)
        v4 = self.fc4(torch.cat([v2, v3], dim=1))
        v5 = self.fc5(v4)
        v6 = self.fc6(v5)
        v7 = self.fc7(v6)
        v8 = self.fc8(v7)
        v9 = self.fc9(torch.cat([v2, v3, v4, v5, v6, v7, v8], dim=1))
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2304)
x2 = torch.randn(1, 2304)
x = torch.cat([x1, x2], dim=1)
