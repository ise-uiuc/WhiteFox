
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(256, 1024, kernel_size=(4, 4), stride=(4, 4), bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1024, 1, 1)
