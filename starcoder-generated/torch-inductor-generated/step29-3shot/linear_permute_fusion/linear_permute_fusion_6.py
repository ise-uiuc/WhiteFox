
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(x1, scale_factor=(1.0, 1.0), mode='nearest')
        v2 = v1.permute(0, 2, 3, 1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
x2 = x1.squeeze(0)
