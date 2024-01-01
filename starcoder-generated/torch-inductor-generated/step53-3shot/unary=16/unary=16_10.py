
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28*28, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.relu()
        return v2

# Inputs to the model
x1 = torch.randn(1, 28*28)
