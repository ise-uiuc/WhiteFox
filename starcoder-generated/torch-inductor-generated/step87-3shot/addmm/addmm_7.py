
# t1 represents output tensor of matrix multipication of two tensors
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.mm(inp, torch.randn(3, 3))
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, torch.randn(3, 3))
        v1 = v1.mul(self.t1)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
