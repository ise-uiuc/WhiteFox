
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.AvgPool2d(2, stride=2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = 0.1666666865348816
        v4 = F.hardsigmoid(v2, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
