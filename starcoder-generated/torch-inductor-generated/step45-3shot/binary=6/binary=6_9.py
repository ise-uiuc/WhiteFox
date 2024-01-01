
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d((1, 1))
        self.linear = torch.nn.Linear(512, 10)
 
    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2 = torch.flatten(v1, 1)
        v3 = self.linear(v2)
        v4 = v3 - 9.0
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256, 4, 4)
