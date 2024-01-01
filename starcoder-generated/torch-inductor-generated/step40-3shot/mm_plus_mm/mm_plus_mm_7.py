
class Model(torch.nn.Module):
    def forward(self, a):
        a0, a1, a2, a3, a4 = torch.split(a, 5, dim=1)
        b0 = torch.stack((a0, a1), 0)
        b1 = torch.stack((a2, a3, a4), 0)
        x = torch.stack((b0, b1), 0)
        return x
# Inputs to the model
input = torch.randn(4, 10)
