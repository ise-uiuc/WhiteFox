
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.cumsum(x1, x2, out=None)
        t2 = torch.cumsum(x1, x2, out=t1)
        t3 = torch.cumsum(x1, x2, out=t2)
        return t3
# Inputs to the model
x1 = torch.randn(8, 8, device='cpu')
x2 = (torch.tensor(0, device=x1.device) - 8) * 2
