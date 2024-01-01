
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input0):
        t0 = torch.mm(input0, input0)
        v1 = torch.cat([t0, t0], 1)
        t3 = torch.cat([v1, v1], 1)
        v10 = torch.cat([t3, t3], 1)
        return v10
# Inputs to the model
input0 = torch.randn(10, 10)
