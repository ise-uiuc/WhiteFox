
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 8)
        self.fc5 = torch.nn.Linear(8, 4)
        self.fc6 = torch.nn.Linear(4, 2)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(v1)
        v3 = self.fc3(v2)
        v4 = self.fc4(v3)
        v5 = self.fc5(v4)
        v6 = self.fc6(v5)
        v7 = v6 + torch.tensor([1., 2.], dtype=torch.float32)
        v8 = torch.nn.functional.relu(v7)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
