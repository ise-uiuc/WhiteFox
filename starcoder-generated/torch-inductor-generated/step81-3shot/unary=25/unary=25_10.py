
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.linear = torch.nn.Linear(150, 150)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v2 = F.relu6(self.linear(x1), 0.0)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 150)
