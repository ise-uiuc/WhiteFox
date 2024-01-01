
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 3)
        self.relu6 = torch.nn.ReLU6()
 
    def forward(slef, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        v3 = self.relu6(v2)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
