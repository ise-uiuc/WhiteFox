
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(in_features=3, out_features=24, bias=True)
 
    def forward(self, x1):
        v1 = self.l(x1)
        v2 = v1 * torch.clamp(torch.min(v1), torch.max(v1), min=0, max=6) + 3
        v3 = v2 / 6
        return v3


# Inputs to the model
x1 = torch.randn(1, 3)
