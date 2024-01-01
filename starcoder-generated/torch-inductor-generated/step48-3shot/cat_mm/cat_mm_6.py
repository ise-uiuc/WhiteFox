
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn6 = torch.nn.BatchNorm2d(20, affine=True, track_running_stats=True)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        for loopVar1 in range(575):
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.nn.functional.relu(self.bn6(v1))
        return torch.cat([v1, v1], 1)
# Inputs to the model
x1 = torch.randn(3, 400)
x2 = torch.randn(400, 5)
