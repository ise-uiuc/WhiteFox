
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input0):
        t0 = torch.mm(input0, input0)
        t0 = t0.unsqueeze(0).repeat_interleave(10, 0)
        t1 = torch.cat([t0, t0, t0, t0, t0, t0, t0, t0, t0, t0], 1)
        return t1
# Inputs to the model
input0 = torch.randn(10, 10)
