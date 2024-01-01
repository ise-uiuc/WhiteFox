
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(1)
        v3 = v2.softmax(dim = -1)
        v4 = torch.nn.functional.dropout(v3, p=0.2)
        v5 = v4.matmul(1)
        v6 = self.conv(v5)
        return v6


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 8, 8)
x2 = torch.randn(1, 8, 8, 8)
y1 = m(x1, x2)

