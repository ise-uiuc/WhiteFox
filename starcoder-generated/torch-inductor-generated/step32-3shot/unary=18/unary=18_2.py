
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(5, 5, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return nn.Sigmoid()(v1)
# Inputs to the model
x1 = torch.zeros(1, 5, 128, 128, 128)
