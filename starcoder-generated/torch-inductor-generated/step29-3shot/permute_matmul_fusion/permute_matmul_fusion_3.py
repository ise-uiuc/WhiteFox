
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def v2(x):
            v2_output = torch.matmul(x.permute(0, 2, 1), x1)
            return v2_output
        self.f = torch.jit.trace(v2, x2)
    def forward(self, x1, x2):
        v1 = self.f(x2)
        v3 = torch.bmm(v1, x2.permute(0, 2, 1))
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
