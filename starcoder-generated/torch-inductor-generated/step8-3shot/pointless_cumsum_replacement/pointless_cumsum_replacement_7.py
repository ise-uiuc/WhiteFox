
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.full([1024, 256], 1, dtype=torch.long, layout=torch.strided, device=torch.device('cpu'), pin_memory=False)
        t2 = t1.to(dtype=torch.uint8)
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(1024, 256, device='cpu')
