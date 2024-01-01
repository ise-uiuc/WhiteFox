
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        d = {}
        c = {}
        d['device'] = torch.device('cuda:0')
        c['device'] = torch.device('cuda:0')
        x4 = torch.abs(x1)
        x5 = x4.neg()
        return x5
# Inputs to the model
x1 = torch.tensor([17, 25, 89, 39, 16, 82, 24, 62, 68, 37], device='cuda:0')
