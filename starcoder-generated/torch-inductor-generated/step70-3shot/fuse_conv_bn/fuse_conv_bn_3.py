
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.norm = torch.nn.BatchNorm2d(2, affine=False)
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(2, 3, kernel_size=(1, 3), stride=(1, 2), padding=0, dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 16, 8)
