
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x1 = torch.cat((x, y), dim=1)
        x2 = torch.cat((x, x1), dim=1)
        x3 = torch.cat((x1, x2), dim=1)
        x4 = torch.cat((x, x3), dim=1)
        x5 = torch.cat((x1, x4), dim=1)
        x6 = torch.cat((x2, x5), dim=1)
        x7 = torch.cat((x3, x6), dim=1)
        x8 = torch.cat((x4, x7), dim=1)
        return x8.view(x8.shape).relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 2)
