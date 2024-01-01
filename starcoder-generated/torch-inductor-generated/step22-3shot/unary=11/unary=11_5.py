
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.clamp_max((torch.conv_transpose2d(x1, torch.ones(3, 3, 3, 3),'same') + 3), 6) / 6
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
