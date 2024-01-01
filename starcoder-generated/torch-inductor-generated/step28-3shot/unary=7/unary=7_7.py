
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(29, 29)
        self.relu = torch.nn.ReLU6()
        self.linear_1 = torch.nn.Linear(29, 29)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * self.relu(v1 + 3)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 29)
