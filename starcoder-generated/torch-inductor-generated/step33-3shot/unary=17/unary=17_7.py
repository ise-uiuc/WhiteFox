
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(3, 64, kernel_size=2, stride=2)
        self.conv_transpose1 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.conv_transpose2 = nn.ConvTranspose2d(16, 4, kernel_size=2, stride=1)
    def forward(self,x):
        x = self.conv_transpose(x)
        x = torch.relu(x)
        x = self.conv_transpose1(x)
        x = torch.relu(x)
        x = self.conv_transpose2(x)
        return x
# Inputs to the model
N, C_in, H_in, W_in = 1, 3, 16, 16
# Input tensor of the model
x = torch.randn(N, C_in, H_in, W_in)
