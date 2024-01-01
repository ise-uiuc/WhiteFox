
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float16
        b['layout'] = torch.sparse
        a['dtype'] = torch.float16
        b['device'] = torch.device('cpu')
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.float16
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float16
        t1 = torch.full([512, 1024], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(512, 1024, device='cpu').to_sparse()
