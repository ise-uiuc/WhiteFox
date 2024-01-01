
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        x = torch.rand(2, 2)
        y = torch.cat([x, x, x])
        if y.size(0) > 1: # Dummy comparison which will be optimized away by the backend
            y = torch.sum(y)
        return y
