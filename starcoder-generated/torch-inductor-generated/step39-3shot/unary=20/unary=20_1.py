
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=0)
        self.layer_2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.layer_1(x1)
        v2 = self.layer_2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 234, 322)
