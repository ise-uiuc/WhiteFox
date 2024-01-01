
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(512, 4, kernel_size=(4, 4), stride=(4, 4), bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(104, 3, 7, 7)
