
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.interpolate(x1, x1.shape[2:3], mode='bilinear')
        return x2
# Inputs to the model
x1 = torch.randn(1, 16, 128, 128)
