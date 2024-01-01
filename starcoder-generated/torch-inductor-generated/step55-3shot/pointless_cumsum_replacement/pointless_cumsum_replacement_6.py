
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.uint8
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.int32
        b['dtype_to'] = torch.float
        b['dtype_from'] = torch.int32
        t1 = torch.full([64, 256], 1, dtype=b['dtype'], pin_memory=False)
        t2 = t1.half()
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(64, 256, device='cuda:0')
