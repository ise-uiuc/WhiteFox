
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(8)
 
    def forward(self, x1):
        v1 = self.bn1(self.linear1(x1))
        v2 = v1.tanh()
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
