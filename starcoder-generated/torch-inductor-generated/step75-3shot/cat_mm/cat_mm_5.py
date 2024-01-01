
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        s = [v, v]
        l = [(None, None, None)] if random() > 0.5 else s
        l = l + s
        return torch.cat([i[2] for i in l], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
