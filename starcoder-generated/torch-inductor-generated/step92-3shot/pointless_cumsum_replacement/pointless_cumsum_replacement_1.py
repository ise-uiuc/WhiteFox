
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.sparse_coo
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.uint8
        a['layout'] = torch.sparse_coo
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.uint8
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.uint8
        t1 = torch.full([20, 50], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(20, 50, device='cuda:0')
