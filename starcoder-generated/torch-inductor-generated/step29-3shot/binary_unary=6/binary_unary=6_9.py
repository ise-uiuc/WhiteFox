
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(48, 16)
 
    def forward(self, x):
        x = self.linear(x)
        x = x - 1
        x = relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 10)
