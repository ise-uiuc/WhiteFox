
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3 * 64 * 64, 10)
 
    def forward(self, x):
        x1 = linear(x)
        x2 = torch.nn.functional.relu(x1)
        x3 = x2 + other
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
