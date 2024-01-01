
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)
        self.upsample = torch.nn.Upsample(scale_factor=1.0, mode='nearest')
    def forward(self, x1):
        v1 = self.upsample(x1)
        v3 = self.relu(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 60, 60)
