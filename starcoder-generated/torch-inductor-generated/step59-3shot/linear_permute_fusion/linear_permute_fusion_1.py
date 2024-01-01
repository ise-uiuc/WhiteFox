
class LinearToReshape(torch.nn.Module):
    def forward(self, t):
        l = t.dim()
        assert l < 3, "not supported type"
        if l == 2:
            return t.reshape(t.size(0), 1, -1)
        elif l == 3:
            return t

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, input):
        t = self.linear(input)
        return LinearToReshape()(t)
# Inputs to the model
input = torch.randn(1, 2, device='cpu')
