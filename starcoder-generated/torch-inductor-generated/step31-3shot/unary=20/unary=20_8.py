
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=1,padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1,output_padding=0,bias=False)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
