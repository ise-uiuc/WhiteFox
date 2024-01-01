
class Model(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(256, 10, bias=False)
        self.linear.weight = nn.Parameter(torch.zeros_like(self.linear.weight))
 
    def forward(self, x1, x2):
        y1 = self.linear(x1)
        y2 = self.linear(x2)
        z1 = y1 - x1
        z2 = y2 - x2
        return z1, z2

# Input to the model
x1 = torch.ones(64, 256)
x2 = x1 / 2.0
