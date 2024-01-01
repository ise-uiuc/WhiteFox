
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = torch.cat((x, x), dim=1)
        y2 = torch.cat((x, x), dim=1)
        y3 = y1 * y2
        x = y3.relu()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
