
class Model(torch.nn.Module):
    def forward(self, a0, a1):
        x0 = torch.mm(a0, a1)
        x1 = torch.mm(a0, a0)
        x2 = torch.mm(a1, a1)
        x3 = x1 + x2
        out = x1 * x3
        return x0 + out
# Inputs to the model
a0 = torch.randn(128, 128)
a1 = torch.randn(128, 128)
