
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.full([2048, 1], 1, dtype=torch.int64, layout=torch.strided, device=torch.device('cpu'), pin_memory=False)
        t2 = torch.clamp(t1, min=-2147483648.0)
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(2048, 1, device='cpu')
