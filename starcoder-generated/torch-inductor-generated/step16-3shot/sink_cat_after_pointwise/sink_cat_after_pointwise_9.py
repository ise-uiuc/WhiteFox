
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x1, x2, x3):
        v1 = torch.cat((x1, x2, x0, x3), dim=1)
        v2 = torch.cat((v1, x1), dim=1)
        y = torch.relu(v2)
        v3 = torch.relu(v1)
        y = torch.tanh(v3)
        return y
# Inputs to the model
x0 = torch.randn(1, 2)
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
x3 = torch.randn(1, 2)
