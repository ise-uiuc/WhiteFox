
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(128, 256)
        self.lin2 = torch.nn.Linear(128, 256)
        self.lin3 = torch.nn.Linear(128, 256)
        self.lin4 = torch.nn.Linear(256, 256)
        self.lin5 = torch.nn.Linear(256, 25)
 
    def forward(self, x1, x2):
        v1 = self.lin1(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.lin2(x2)
        v4 = torch.nn.functional.relu(v3)
        v5 = self.lin3(v2)
        v6 = torch.nn.functional.relu(v5)
        v7 = self.lin4(v4)
        v8 = torch.nn.functional.relu(v7)
        v9 = v6 + v8
        v10 = self.lin5(v9)
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 128)
x2 = torch.randn(4, 128)
