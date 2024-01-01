
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input0):
        t = torch.cat([], 0)
        for _ in range(10):
            t = torch.mm(input0, input0)
        return t
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
