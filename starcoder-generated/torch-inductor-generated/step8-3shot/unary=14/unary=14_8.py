
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtransp_relu = torch.nn.Sequential(*[
        torch.nn.ConvTranspose2d(3, 8, 2, stride=2),
        torch.nn.ReLU(),
        ])
    def forward(self, x1):
        v1 = self.convtransp_relu(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
