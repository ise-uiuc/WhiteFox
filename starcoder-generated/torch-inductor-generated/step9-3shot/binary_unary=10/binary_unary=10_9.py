
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        y = v1 + x2
        y = torch.relu(y)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 10)
x2 = torch.randn(1, 10, 10)

