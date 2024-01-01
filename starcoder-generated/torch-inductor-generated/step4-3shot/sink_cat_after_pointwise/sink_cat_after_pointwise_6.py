
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = x.view(x.shape[0], x.shape[1], x.shape[2] // 4, -1)
        b = a.view(a.shape[0], a.shape[1], a.shape[2], a.shape[3] * 4)
        b = torch.abs(b)
        c = torch.sum(b, dim=2)
        return c
# Inputs to the model
x = torch.randn(2, 3, 10)
