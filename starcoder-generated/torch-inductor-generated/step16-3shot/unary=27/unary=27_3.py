
class Model(nn.Module):
    def __init__(self, m0=1.5):
        super().__init__()
        self.a = torch.nn.Parameter(m0)
    def forward(self, input):
        v1 = input.clamp_max(self.a)
        v2 = input.clamp_min(self.a)
        return v1, v2
# Inputs to the model
input = torch.randn(1, 1, 64, 64)
