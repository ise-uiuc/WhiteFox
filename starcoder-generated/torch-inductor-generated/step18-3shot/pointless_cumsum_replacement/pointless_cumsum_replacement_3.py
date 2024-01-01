
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        a['dtype'] = torch.uint8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        b['dtype'] = torch.short
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.uint8
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float32
        t1 = torch.full([256, 1024], 1, dtype=a['dtype'], layout=a['layout'], device=a['device'], pin_memory=False)
        t2 = t1.to(dtype=b['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.tensor([0.5132, -1.2458, 0.9574, -0.438, -1.545, -1.1948, 0.7477, -0.3177, 1.3193, 0.4853], device='cuda:0')
