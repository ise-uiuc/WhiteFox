
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - x1
        return v2

# Initializing the model
m2 = Model2()

# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
