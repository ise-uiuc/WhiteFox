
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x1):
        v0 = x.reshape(-1, 1, 2, 2).squeeze()
        v0 = x.reshape(-1, 1, 2, 2).squeeze()
        v0 = x1.reshape(-1, 1, 2, 2).squeeze()
        v0 = x1.reshape(-1, 1, 2, 2).squeeze()
        return torch.bmm(v0, v0)
# Inputs to the model
x = torch.randn(2, 4)
x1 = torch.randn(1, 2, 2)
