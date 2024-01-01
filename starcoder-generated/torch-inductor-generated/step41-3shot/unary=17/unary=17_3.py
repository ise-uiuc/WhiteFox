
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(2, 24, kernel_size=[2, 2], stride=(2, 2), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.relu(v1)
        v3 = v2.transpose(1, 3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2, 4)
