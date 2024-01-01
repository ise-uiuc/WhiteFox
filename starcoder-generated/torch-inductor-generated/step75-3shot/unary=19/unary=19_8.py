
class Model(torch.nn.Module):
    def __init__(self, w1=0.3, w2=0.4, b=-0.2):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
 
        w = torch.nn.Parameter(torch.tensor([w1, w2], dtype=torch.float))
        b = torch.nn.Parameter(torch.tensor([b], dtype=torch.float))
        self.linear.weight = w
        self.linear.bias = b
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
__w1__ = 0.3
__w2__ = 0.4
__b__ = -0.2

m = Model(__w1__, __w2__, __b__)

# Inputs to the model
x1 = torch.randn(1, 8)
