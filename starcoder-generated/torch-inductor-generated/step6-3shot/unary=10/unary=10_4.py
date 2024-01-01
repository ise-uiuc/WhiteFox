
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        l1 = self.conv1(x1)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        l6 = self.conv2(l5)
        return l6

# Initializing model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
l7 = m(x1)

