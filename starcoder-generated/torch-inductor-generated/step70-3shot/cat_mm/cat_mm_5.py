
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x = {}
        x[str(0)+str(0)+str(0)+str(0)] = torch.cat([x1, x2], 1)
        x[str(0)+str(0)+str(0)+str(1)] = torch.cat([x[str(0)+str(0)+str(0)+str(0)], x[str(0)+str(0)+str(0)+str(0)]], 1)
        x[str(0)+str(0)+str(1)+str(0)] = torch.cat([x[str(0)+str(0)+str(0)+str(1)], x[str(0)+str(0)+str(0)+str(1)]], 1)
        x[str(0)+str(0)+str(1)+str(1)] = torch.cat([x[str(0)+str(0)+str(1)+str(0)], x[str(0)+str(0)+str(1)+str(0)]], 1)
        x[str(1)+str(0)+str(0)+str(0)] = torch.cat([x[str(0)+str(0)+str(1)+str(1)], x[str(0)+str(0)+str(1)+str(1)]], 1)
        x[str(1)+str(0)+str(0)+str(1)] = torch.cat([x[str(1)+str(0)+str(0)+str(0)], x[str(1)+str(0)+str(0)+str(0)]], 1)
        x[str(1)+str(0)+str(1)+str(0)] = torch.cat([x[str(1)+str(0)+str(0)+str(1)], x[str(1)+str(0)+str(0)+str(1)]], 1)
        x[str(1)+str(0)+str(1)+str(1)] = torch.cat([x[str(1)+str(0)+str(1)+str(0)], x[str(1)+str(0)+str(1)+str(0)]], 1)
        x[str(1)+str(1)+str(0)+str(0)] = torch.cat([x[str(1)+str(0)+str(1)+str(1)], x[str(1)+str(0)+str(1)+str(1)]], 1)
        x[str(1)+str(1)+str(0)+str(1)] = torch.cat([x[str(1)+str(1)+str(0)+str(0)], x[str(1)+str(1)+str(0)+str(0)]], 1)
        x[str(1)+str(1)+str(1)+str(0)] = torch.cat([x[str(1)+str(1)+str(0)+str(1)], x[str(1)+str(1)+str(0)+str(1)]], 1)
        x[str(1)+str(1)+str(1)+str(1)] = torch.cat([x[str(1)+str(1)+str(1)+str(0)], x[str(1)+str(1)+str(1)+str(0)]], 1)
        return x[str(1)+str(1)+str(1)+str(1)]
# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(2, 4)
