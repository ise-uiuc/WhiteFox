
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.float32
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float16
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float32
        a['dtype_from'] = torch.float16
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float16
        t1 = torch.randn([1048576, 2048], device=b['device'])
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 0)
        return t3
# Inputs to the model
x1 = torch.randn(1048576, 2048, device='cuda:0')
