
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, bias):
        # y = torch.cat((x, x + 1.0), dim=1)
        y = torch.cat((x, bias), dim=1)
        y = torch.relu(y)
        return y
# Inputs to the model
x = torch.randn(2, 2, 2)
bias = torch.randn(2, 1)
