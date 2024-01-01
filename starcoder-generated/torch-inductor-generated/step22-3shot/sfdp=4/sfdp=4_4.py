
class Model(torch.nn.Module):
    def __init__(self, n1, n2, n3):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, n1, 1, stride=1, padding=0)
        self.c2 = torch.nn.Conv2d(n1, n2, 3, stride=1, padding=1)
        self.c3 = torch.nn.Conv2d(n2, n3, 3, stride=1, padding=1)
 
    def forward(self, x1):
        x2 = self.c1(x1)
        x3 = self.c2(x2)
        x4 = self.c3(x3)
        return x4

# Initializing the model
n1 = 4
n2 = 8
n3 = 16
m = Model(n1, n2, n3)

# Input for the model
x1 = torch.randn(1, 3, 64, 64)
