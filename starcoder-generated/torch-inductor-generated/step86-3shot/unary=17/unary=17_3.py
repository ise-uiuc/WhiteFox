
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranpose = torch.nn.ConvTranspose2d(16, 16, 3, stride=1)
    def forward(self, x1):
        v1 = self.convtranpose(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = v2.transpose(3, 2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 24, 24)
