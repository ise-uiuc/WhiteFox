
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.k = torch.nn.Conv2d(32, 64, 1, stride=1)
        self.v = torch.nn.Conv2d(32, 64, 1, stride=1)
        self.scale_factor = math.sqrt(64)
 
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        s = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        d = torch.nn.functional.dropout(s, 0.5)
        o = torch.matmul(d, v)
        return o

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 32, 16, 16)
