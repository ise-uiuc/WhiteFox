
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
        self.a = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
 
    def forward(self, x1, x2):
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)
        a = self.a(x2)
        b = k.transpose(-2, -1) / q.size(-1)
        v = torch.softmax((q @ b + a), dim=-1) @ v
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
