
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=False)
 
    def forward(self, x1):
       v1 = self.linear(x1)
       v2 = v1 + x1
       return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
