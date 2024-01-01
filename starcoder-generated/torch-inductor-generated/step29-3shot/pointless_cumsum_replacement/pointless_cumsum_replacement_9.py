
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.cuda.FloatTensor
        b['layout'] = torch.strided
        b['device'] = torch.device('cpu')
        a['dtype'] = torch.cuda.FloatTensor
        a['layout'] = torch.strided
        a['device'] = torch.device('cpu')
        a['dtype_to'] = torch.cuda.FloatTensor
        a['dtype_from'] = torch.cuda.FloatTensor
        b['dtype_to'] = torch.cuda.FloatTensor
        b['dtype_from'] = torch.cuda.FloatTensor
        t1 = torch.full([65632, 3840], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3
# Inputs to the model
x1 = torch.randn(65632, 3840, device='cpu')
