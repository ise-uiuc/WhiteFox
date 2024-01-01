
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.unsqueeze(x, 1)
        x2 = torch.unsqueeze(x1, 2)
        x3 = torch.cat((x, x, x, x), 1)
        x4 = torch.reshape(x3, (1, 4, 8, 32))
        x5 = torch.reshape(x, (1, 1, 4, 33))
        x6 = torch.cat((x5, x4, x5), 1)
        x7 = torch.cat((x2, x2, x6, x4, x4), 2)
        return x7
# Inputs to the model
x = torch.randn(1, 4, 8, 32)
