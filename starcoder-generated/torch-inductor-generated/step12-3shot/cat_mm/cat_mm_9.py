
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        t = [v1] * 50
        return torch.cat(t, 1)
# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 4)
