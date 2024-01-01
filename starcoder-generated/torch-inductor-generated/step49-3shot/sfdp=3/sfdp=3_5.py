
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkvconv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        k1 = self.qkvconv(x1)
        k2 = self.qkvconv(x2)
        k3 = k1.transpose(-2, -1)
        v1 = k1 * 0.125
        v2 = k2.transpose(-2, -1) * 0.125
        v3 = torch.matmul(v1, v2)
        v4 = v3 * 0.25
        v5 = v3 * 0.5
        v6 = torch.exp(v5)
        v7 = torch.nn.functional.dropout(v6, p=0.25)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
x2 = torch.randn(1, 64, 64, 64)
