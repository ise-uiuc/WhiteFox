
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.tanh(x * x)
        b = torch.relu(x + y)
        c = torch.sigmoid(y * y)
        d = torch.tanh(b + c)
        e = torch.relu(-b + d)
        f = torch.sigmoid(d * d)
        return torch.cat((y, b, c * c), dim=1)
# Inputs to the model
x = torch.randn(1, 2, 4)
