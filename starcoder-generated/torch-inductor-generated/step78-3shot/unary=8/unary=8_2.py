
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(4, 14, 4, 2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(14, 20, 4, 2)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(20, 26, 4, 2)
        self.conv_transpose_last = torch.nn.ConvTranspose2d(26, 32, 4, 2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v1_add = v1 + 3
        v1_clamp = torch.clamp(v1_add, min=0)
        v1_relued = torch.relu(v1_clamp)
        v2 = self.conv_transpose2(v1_relued)
        v3 = self.conv_transpose3(v2)
        v4 = self.conv_transpose_last(v3)
        return v4
# Inputs to the model
x1 = torch.randn(8, 4, 255, 255)
