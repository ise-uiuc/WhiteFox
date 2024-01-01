
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(3, 32, 3, padding=1, stride=2)
    def forward(self, x_in):
        x_in = x_in.permute(0, 4, 3, 2, 1)
        l1 = torch.relu(self.conv_transpose(x_in))
        return l1
# Inputs to the model
x_in = torch.randn(1, 8, 128, 128, 3)
