
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv_transpose2 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=(2,1,3), padding=(0,1,3))
        self.relu = nn.ReLU()
    def forward(self, x1):
        return torch.relu(self.conv_transpose1(x1) + self.conv_transpose2(self.conv_transpose1(x1)))
# Inputs to the model
x1 = torch.randn(2,2,2,1,8,7)
