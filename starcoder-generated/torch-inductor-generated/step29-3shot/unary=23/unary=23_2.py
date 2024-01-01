
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 3, (2, 5), stride=(1, 3), padding=(1, 1))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 4, (4, 2), stride=(4, 1), padding=(2, 2))
        self.conv_transpose3 = torch.nn.ConvTranspose2d(5, 6, kernel_size=1, stride=1, padding=0)
    def forward(self, x): 
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        x = self.conv_transpose3(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 5, 7)
