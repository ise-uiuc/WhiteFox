
class Model(torch.nn.Module):
    def __init__(self, n1, n2):
        super().__init__()
        self.linear = torch.nn.Linear(n1, n2)
 
    def forward(self, x):
        v1 = self.linear(x)
        return v1 + self.some_tensor

# Initializing the model
m = Model(5, 4)

# Inputs to the model
x1 = torch.randn(1, 5)
