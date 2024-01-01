
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x0 = torch.nn.functional.pad(x1, pad=(3, 3, 3, 3))
        x3 = torch.nn.functional.relu(x0)
        x4 = torch.nn.functional.conv2d(x3, weight=x2, stride=(1, 1), padding=(3, 3))
        x5 = x4 + x1
        x6 = torch.nn.functional.relu(x5)
        l0 = torch.nn.functional.padding(x1, pad=(3, 3, 3, 3))
        l3 = torch.nn.functional.relu(l0)
        l4 = torch.nn.functional.conv2d(l3, weight=x2, stride=(1, 1), padding=(3, 3))
        l5 = l4 + x1
        l6 = torch.nn.functional.relu(l5)
        y = torch.nn.functional.conv2d(l6, weight=x2, stride=(1, 1), padding=(1, 1))
        return y
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 3, 3)
