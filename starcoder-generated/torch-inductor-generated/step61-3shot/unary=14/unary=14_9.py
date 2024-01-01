
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()	
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                       bias=False)
    def forward(self, x):
        v = self.conv_transpose_2(x)
        return v
# Inputs to the model
x = torch.randn(1, 64, 42, 23)
