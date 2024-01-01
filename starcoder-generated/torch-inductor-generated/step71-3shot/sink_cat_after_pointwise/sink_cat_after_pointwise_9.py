
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, w, b):
        x = torch.cat([x, x], dim=1)
        x = x.tanh()
        x = x.permute(0, 2, 1)
        x = x.tanh()
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
w = torch.randn(4, 1)
b = torch.randn(1, 1, 1)
