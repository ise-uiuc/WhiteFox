
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = []
        v1 = torch.cat([v1, v1], -1)
        v1 = torch.cat([v1, v1], -1)
        return v1 + torch.ones([1, 7])
# Inputs to the model
x1 = torch.randn(1, 1)
