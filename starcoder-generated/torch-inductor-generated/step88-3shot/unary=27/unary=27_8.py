
class Model(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.min = min
    def forward(self, x1):
        v1 = torch.clamp_min(x1, self.min)
        return v1
min = 0.26526717315100344
# Inputs to the model
x1 = torch.randn(4, 8)
