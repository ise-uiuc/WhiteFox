
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = F.relu(x1)
        return torch.cat([v1, v1, v1], 1)
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(4, 100)
