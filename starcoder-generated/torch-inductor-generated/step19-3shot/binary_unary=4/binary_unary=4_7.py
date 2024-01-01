
class Model(torch.nn.Module):
    def __init__(self, weight=torch.tensor([[-1.2, 4.2, 5.2, 0.2], [2.5, 2.4, 10.0, 1.8], [-1.5, -0.5, 2.7, -3.0], [1.2, -3.2, 1.5, 1.3]])):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.linear.weight = torch.nn.Parameter(weight)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + y
        v3 = nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
