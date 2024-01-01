
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.ones(([128, 1000], device='cuda:0'))
        t2 = torch.cumsum(t1, 1)
        return t2
# Inputs to the model
x1 = torch.randn(128, 1000, device='cuda:0')
