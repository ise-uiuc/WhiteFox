
class LinearWithOtherTensorAddition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = LinearWithOtherTensorAddition()

# Inputs to the model
x1 = torch.randn(3, 10)
