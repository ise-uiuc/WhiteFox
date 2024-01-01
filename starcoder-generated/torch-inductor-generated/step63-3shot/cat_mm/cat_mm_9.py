
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        for i in range(10):
            v = torch.mm(x1, x2)
            x1.add_(i + x2[0].item())
            x2.add_(v)
        return torch.cat([x1, x2], -1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
