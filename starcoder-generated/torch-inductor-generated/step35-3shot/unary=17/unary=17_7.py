
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d_1 = torch.nn.ConvTranspose2d(8, 16, (1, 1), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose2d_1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 8, 16, 16) 
