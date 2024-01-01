
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 6)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear1(x1)
        x2 = None
        if 'x2' in kwargs:
            x2 = kwargs['x2']
            v2 = v1 + x2
        