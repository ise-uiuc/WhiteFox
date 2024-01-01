
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 32, kernel_size = (3,3,3), padding=(1,1,1))
    def forward(self, x):
        x = torch.relu(self.conv(x))
        return x
# Inputs of the model
x = torch.randn(2, 1, 64, 64, 64)
