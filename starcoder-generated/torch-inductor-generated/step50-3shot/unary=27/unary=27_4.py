
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = torch.clamp_min(x1, self.min)
        v2 = torch.clamp_max(v1, self.max)
        return v2
min = 0.46075920000743866
max = 0.8271828066825867
# Inputs to the model
x1 = torch.randn(1, 1, 64, 48)
