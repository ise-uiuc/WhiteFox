
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = F.relu6(self.linear(x1) + 3).clamp(_min=0.0, _max=6.0)
        v2 = v1 / 6.0
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
