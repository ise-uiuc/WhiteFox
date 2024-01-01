
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 1, 9, 2, 2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1, 5, 4, 2, 2)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(5, 25, 3, 2, 2)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(25, 100, 2, 2, 2)
        self.conv_transpose5 = torch.nn.ConvTranspose2d(100, 100, 1, 2, 2)
    def forward(self, x1, x2, x3, x4):
        v11 = self.conv_transpose1(x1)
        v8 = self.conv_transpose2(x2)
        # This ConvTranspose2d layer has output_padding = 1
        v10 = self.conv_transpose3(self.conv_transpose4(self.conv_transpose5(v11))) # Apply fused operations 
        v9 = v10 + 3
        v12 = torch.clamp(v9, min=0)
        v13 = torch.clamp(v12, max=6)
        v26 = v8 * v13
        v27 = v26 / 6
        return v27
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
x2 = torch.randn(1, 1, 112, 112)
x3 = torch.randn(1, 1, 56, 56)
x4 = torch.randn(1, 2, 28, 28)
