
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu = torch.nn.Linear(10, 100)
 
    def forward(self, x1):
        x2 = self.linear_relu(x1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100, 10)
