
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_0 = torch.nn.Sequential(torch.nn.ConvTranspose2d(8, 8, [3, 5], stride=[1, 2], padding=1, dilation=2, output_padding=2), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.module_0(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 8, 128, 9)
