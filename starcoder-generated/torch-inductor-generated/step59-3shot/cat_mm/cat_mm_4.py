
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        return torch.cat(20*[v1])
# Inputs to the model
x1 = torch.randn(32, 224)
x2 = torch.randn(224, 1)
