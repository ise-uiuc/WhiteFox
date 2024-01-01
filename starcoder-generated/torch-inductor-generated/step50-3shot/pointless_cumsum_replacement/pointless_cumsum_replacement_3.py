
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(13, 1)
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int8
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.int8
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int8
        a['dtype_from'] = torch.int8
        b['dtype_to'] = torch.int8
        b['dtype_from'] = torch.int8
        t1 = torch.full([10240, 13], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return self.fc1(t3)
# Inputs to the model
x1 = torch.randn(10240, 13, device='cuda:0')
