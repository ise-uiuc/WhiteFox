
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1 * x2
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 3)
