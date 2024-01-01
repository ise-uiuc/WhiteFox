
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t3 = torch.ones([1, 2], dtype=torch.float64, device='cuda:0', pin_memory=False)
    def forward(self, x1):
        t0 = torch.full([1, 4], 1, dtype=torch.float64, layout=torch.strided, device='cuda:0', pin_memory=False)
        t1 = self.t3.to(dtype=torch.float64)
        t2 = torch.cumsum(t1, 1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 2, device='cuda:0')
