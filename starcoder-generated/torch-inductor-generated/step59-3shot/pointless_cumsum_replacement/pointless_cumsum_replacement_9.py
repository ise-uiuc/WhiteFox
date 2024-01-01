
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.full([256, 256], 1, dtype=torch.float16, layout=torch.strided, device=torch.device('cuda:0'), pin_memory=False)
        t2 = t1.contiguous()
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(256, 256, device='cuda:0')
