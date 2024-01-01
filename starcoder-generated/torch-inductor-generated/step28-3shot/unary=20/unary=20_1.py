
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, padding=(2, 2), kernel_size=(4, 2), stride=(2, 5))
    def forward(self, x):
        v1 = torch.sigmoid(self.conv2d(x))
        return v1
# Input to the model
x1 = torch.rand(1,3,3,4)
