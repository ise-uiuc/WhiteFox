
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.full([4708, 16], 1, dtype=torch.float16, layout=torch.strided, device=torch.device('cuda:0'), pin_memory=False)
        t1 = t1.to(dtype=torch.float32)
        t2 = torch.cumsum(t1, 1)
        return t2
# Inputs to the model
x1 = torch.randn(4708, 16, device='cuda:0')
