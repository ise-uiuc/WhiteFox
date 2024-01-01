
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = torch.cat((x, x), dim=1).view(-1, 2)
        x1, x2 = y1[:, :2], y1[:, 2:]
        x = x1.tanh() + x2.sigmoid()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
