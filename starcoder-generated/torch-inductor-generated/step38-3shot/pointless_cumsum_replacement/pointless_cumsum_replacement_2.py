
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.int64
        a['dtype'] = torch.int64
        t1 = b['dtype'].to(a['dtype'])
        t2 = t1.bool()
        return t2
# Inputs to the model
x1 = None
