
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forwward(self, x1, x2, x3):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        return torch.cat([v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1, v1], 1)
# Inputs to the model
# x1 = torch.randn(1, 15)
# x2 = torch.randn(15, 31)
# x3 = torch.randn(1, 15)
