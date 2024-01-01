
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch Lite doesn't support `padding` as an argument for Conv2d.
        # This is a workaround.
        self.conv_2 = torch.nn.Conv2d(1, 1, 23, bias=False,)
        self.conv_transpose = torch.nn.ConvTranspose2d(1,1,8,(5))
    def forward(self, x1):
        x2 = F.relu(self.conv_2(x1))
        x3 = F.relu6(x2 + 3)
        x4 = self.conv_transpose(x3)
        return x4
# Input to the model
x1 = torch.randn(1, 1, 25, 25)
