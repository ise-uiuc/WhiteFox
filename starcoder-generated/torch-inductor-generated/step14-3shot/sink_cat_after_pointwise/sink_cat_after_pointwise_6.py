
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x], dim=1)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 4)
