
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = x
        b = x
        x = torch.cat([a, b], dim=1)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
