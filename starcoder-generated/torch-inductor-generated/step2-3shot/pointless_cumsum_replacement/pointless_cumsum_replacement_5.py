
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.uint8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.uint8
        b['dtype_to'] = torch.int32
        b['dtype_from'] = torch.float16
        t1 = torch.full([1048576], -1.0, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(device=a['device'], dtype=a['dtype'])
        t3 = t2.to(dtype=b['dtype_to'])
        return t3
# Inputs to the model
x1 = torch.randn(1048576, device='cuda:0')
