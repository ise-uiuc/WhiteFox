
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Input tensor
x1 = torch.randn(2)
x2 = torch.randn(3)

# Output of the model
