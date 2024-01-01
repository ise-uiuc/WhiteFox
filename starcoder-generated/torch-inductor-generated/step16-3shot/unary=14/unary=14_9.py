
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvTranspose_pointwise = torch.nn.ConvTranspose2d(4, 16, (3, 4), stride=(2, 3), padding=(0, 1), dilation=2, output_padding=1)
        self.BatchNorm2d = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()
        self.ReLU_fusion = torch.nn.RReLU(0.1, 0.3)
    def forward(self, x1):
        v1 = self.ConvTranspose_pointwise(x1)
        v2 = self.BatchNorm2d(v1)
        v3 = self.relu(v2)
        v4 = self.ReLU_fusion(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
