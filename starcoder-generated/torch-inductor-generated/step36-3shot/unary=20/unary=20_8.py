
class Model(torch.nn.Module):
    def __init__(self, in_channel, out_channel, filter_size=40, stride=1):
        super().__init__()
        self.conv_transpose = torch.nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=[1,2,3], stride=1, bias=True, padding=1)
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.sigmoid(v1)        
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 30, 60, 40)
