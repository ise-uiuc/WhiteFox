
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(34, 128, 1)
        self.conv2 = torch.nn.Conv2d(6,20,3)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(2,34,2,stride=2,output_padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(34, 2, 3, output_padding=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(4,3,2,padding=1,output_padding=1,bias=0)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv_transpose_1(x)
        x = self.conv_transpose_2(x)
        x = self.conv_transpose_3(x)
        return x,y
# Inputs to the model
x = torch.randn(1,6,32, 32)
y = torch.randn(1,2, 4, 4)
