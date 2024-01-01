
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.exp(torch.neg(x1))
        v2 = torch.mul(v1, x1)
        v3 = torch.sin(v2)
        v4 = torch.mul(v3, v3)
        v5 = torch.asin(v4)
        v6 = torch.asinh(v1)
        v7 = torch.tanh(v6)
        return torch.sigmoid(-v2 * v5 * v7 * 0.4881181979645516355674595144914928437215305248016347967764379097)
# Inputs to the model
x1 = (torch.randn(8, 3, 512, 728) - 0.5) * 10
