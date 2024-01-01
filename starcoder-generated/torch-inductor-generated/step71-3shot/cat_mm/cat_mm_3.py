
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        for loopVar1 in range(32173):
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            for i in range(2, 51):
                v1 = torch.mm(x1, x2)
                v1 = torch.mm(x1, x2)
                v1 = torch.mm(x1, x2)
                v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        for loopVar1 in range(6417):
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
        return torch.cat([v1, v2], 1)
# Inputs to the model
x1 = torch.randn(1, 89)
x2 = torch.randn(89, 89)
