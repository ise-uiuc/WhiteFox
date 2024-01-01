
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x.view(4, -1), x.view(4, -1), x.view(4, -1)])
        x1 = torch.matmul(y, y.permute([1, 0]))
        x = torch.sum(x1, dim=0, keepdim=False)
        return x.view(y.shape[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 1, 4)
