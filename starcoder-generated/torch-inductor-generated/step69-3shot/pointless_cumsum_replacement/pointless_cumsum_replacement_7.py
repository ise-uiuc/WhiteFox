
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x3, x2):
        b = {}
        a = {}
        b['dtype'] = torch.int64
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a = {}
        a['dtype'] = torch.int64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int64
        a['dtype_from'] = torch.int64
        b['dtype_to'] = torch.int64
        b['dtype_from'] = torch.int64
        t1 = x2.to(device="cuda:0")
        t2 = torch.div(t1, x3, rounding_mode="floor")
        t1 = t2.to(device=b['device'])
        t3 = torch.ceil(t1)
        return t3
# Inputs to the model
x3 = 0.001
x2 = torch.randn(17, 64, device='cuda:0')
