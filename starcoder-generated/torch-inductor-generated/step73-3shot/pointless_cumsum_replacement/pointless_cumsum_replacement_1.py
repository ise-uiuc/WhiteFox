
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = torch.uint8 
        a = torch.int8 
        b = torch.uint8 
        a = torch.int8 
        t1 = torch.full([16384], 1, dtype=b, layout=torch.strided, device=torch.device('cuda:0'), pin_memory=False)
        t2 = t1.to(dtype=a)
        t3 = torch.cumsum(t2, 0)
        return t3
# Inputs to the model
x1 = torch.randn(16384, device='cuda:0')
