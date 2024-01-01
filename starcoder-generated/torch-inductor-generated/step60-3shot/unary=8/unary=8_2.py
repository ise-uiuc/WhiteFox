
class ExampleModel(torch.nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(14, 10, kernel_size=3, padding=0, stride=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 6, kernel_size=5, padding=2, stride=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(6, 2, kernel_size=3, padding=0, stride=2)
        # This is an uncommon argument
        self.conv_transpose3.return_indices = True
        # This is an uncommon argument
        self.conv_transpose3.dilation = 2
        self.conv_transpose4 = torch.nn.ConvTranspose2d(2, 3, kernel_size=5, padding=0, stride=2)
    def forward(self, x):
        v1 = self.conv_transpose1(x)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        # Example of an uncommon argument
        v3 = v3[0]
        v4 = self.conv_transpose4(v3)
        return v4
# Inputs to the model
x = torch.rand(16, 14, 50, 50)
