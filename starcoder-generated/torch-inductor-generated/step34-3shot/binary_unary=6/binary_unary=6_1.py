
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, x):
        x1 = self.linear(x)
        x2 = x1 - 0.5
        x3 = torch.relu(x2)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
