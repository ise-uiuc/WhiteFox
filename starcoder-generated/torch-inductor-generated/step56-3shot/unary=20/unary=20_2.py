
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sigmoid(torch.nn.functional.conv_transpose2d(x, torch.nn.init.kaiming_normal_(torch.empty(x.size()[1], (x.size()[1] * 2), kernel_size=(2, 2), stride=(2, 2)), mode='fan_out'), bias=None, stride=(2, 2), padding=(0, 0), dilation=(1, 1)))
# Inputs to the model
x = torch.randn(3, 2, 10, 20)
