
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.other = torch.Tensor([3, -1, 4, 100])
 
    def forward(self, x):
        x1 = self.linear(x)
        x2 = x1 - self.other
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(4)
