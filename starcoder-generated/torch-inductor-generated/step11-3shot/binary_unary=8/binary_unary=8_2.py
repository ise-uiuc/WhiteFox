
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x1):
        x2 = x1 + 1
        x3 = torch.relu(x2)
        x4 = x1 + 1
        x5 = torch.relu(x4)
        x6 = x3 + x5
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
