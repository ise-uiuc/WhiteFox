
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(6, 1, kernel_size=(2,3),(1,2),(0,1),(1,-1),bias=True)
    def forward(self, x):
        return self.conv(x)
# Inputs to the model
x = torch.randn(1, 6, 5, 5)
