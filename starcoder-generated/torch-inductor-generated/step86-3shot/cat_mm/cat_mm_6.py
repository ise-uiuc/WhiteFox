
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        for loopVar3 in range(10):
            v = torch.cat([v, torch.mm(x1, x2).unsqueeze(0)], 0)
            v = torch.cat([v, torch.mm(x2, x2).unsqueeze(0)], 0)
            v = torch.cat([v, torch.mm(x3, x2).unsqueeze(0)], 0)
            v = torch.cat([v, torch.mm(x4, x2).unsqueeze(0)], 0)
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(2, 1)
x2 = torch.randn(1, 1)
