
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, X1, X2, X3, X4, X5, X6):
        x1 = torch.cat([X1, X2, X3],dim=0)
        x2 = torch.cat([X4, X5, X6],dim=0)
        y = torch.cat([x1, x2], dim=0)
        x = y.view(y.shape[0], y.shape[1], y.shape[2], 1, 1, -1).flatten()
        x = x[::7].view(x.shape[0], x.shape[1], x.shape[2], -1).relu().flatten()
        return x[::13].view(x.shape[0], x.shape[1], x.shape[2], -1)
# Inputs to the model
X1 = torch.randn(1, 16, 14, 12)
X2 = torch.randn(1, 16, 7, 8)
X3 = torch.randn(8, 16, 14, 12)
X4 = torch.randn(8, 16, 7, 8)
X5 = torch.randn(6, 16, 14, 12)
X6 = torch.randn(6, 16, 7, 8)
