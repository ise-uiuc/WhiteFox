
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v = self.linear(x)
        sigmoid = torch.nn.Sigmoid()
        v = sigmoid(v)
        v = v * v
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
