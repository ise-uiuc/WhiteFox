
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(40, 20)
        self.linear_1 = torch.nn.Linear(20, 20)
        self.linear_2 = torch.nn.Linear(20, 20)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        v3 = self.linear_1(v2)
        v4 = torch.relu(v3)
        v5 = self.linear_2(v4)
        v6 = torch.relu(v5)
        return v1, v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 40)
