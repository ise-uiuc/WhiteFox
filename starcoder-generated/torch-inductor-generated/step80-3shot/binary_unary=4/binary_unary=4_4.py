
class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1, x2):
        y1 = self.linear(x1)
        y2 = y1 + x2
        y3 = torch.relu(y2)
        return y3

# Initializing the model
m = Model(torch.ones(10, 10))

# Inputs to the model
x1 = torch.randn(20, 10)
x2 = torch.randn(20, 10)
