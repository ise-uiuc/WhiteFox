
class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, **kwargs):
        t = x + kwargs['key']
        return t
class M2(M1):
    pass
class M3(M2):
    pass
class M4(M3):
    pass
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
kwargs = {'key': torch.randn(1, 16, 64, 64)}
