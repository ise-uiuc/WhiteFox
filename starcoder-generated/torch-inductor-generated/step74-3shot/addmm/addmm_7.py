
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(1, 10)
    def forward(self, inp):
        return torch.clamp(torch.mm(self.inp, inp))
