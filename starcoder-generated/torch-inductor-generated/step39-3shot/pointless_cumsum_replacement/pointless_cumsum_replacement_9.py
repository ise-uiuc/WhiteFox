
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b = {}
        t1 = torch.full([2048, 1], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(device=b['device'])
        return t2
# Inputs to the model
x1 = torch.randn(2048, 1, device='cuda:0')
