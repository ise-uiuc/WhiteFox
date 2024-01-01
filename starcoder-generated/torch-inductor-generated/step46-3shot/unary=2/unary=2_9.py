
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 1, 1, stride=(1, 1), padding=(0, 0))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(2, 1, 3, stride=(2, 2), padding=(0, 0))
        self.conv_transpose3 = torch.nn.ConvTranspose2d(1, 2, 2, stride=(1, 3), padding=(1, 2))
        self.conv2d = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv1d = torch.nn.Conv1d(1, 1, 1, stride=(1), padding=(0))
    def forward(self, x1):
        v2 = self.conv_transpose1(x1)
        v1 = self.conv_transpose2(v2)
        v4 = self.conv_transpose3(v1)
        v3 = self.conv2d(v4)
        v5 = self.conv1d(v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
