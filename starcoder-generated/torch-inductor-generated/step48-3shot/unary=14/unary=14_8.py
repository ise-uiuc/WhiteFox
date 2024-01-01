
class toto(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.functional.pad(torch.empty(507904, 768, 26, 26, dtype=torch.float), (26, 26, 26, 26), value=0.0)
    def forward(self, x):
        v1 = self.pad(x)
        return v1
# Inputs to the model
x1 = torch.randn(1, 59712, 56, 56)
x2 = torch.randn(1, 768, 26, 26)
