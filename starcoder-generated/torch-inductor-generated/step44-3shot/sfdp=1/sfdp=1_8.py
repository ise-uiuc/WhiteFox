
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = torch.matmul(v1, x2.transpose(-2, -1))
        v3 = v2.div(0.5)
        v4 = v2.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.25)
        v6 = v3 * v5
        m  = torch.matmul(v6, x3)
        return m

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
x3 = torch.randn(1, 5, 64, 64)
r  = m(x1, x2, x3)

