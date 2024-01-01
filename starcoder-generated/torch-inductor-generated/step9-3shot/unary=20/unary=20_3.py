
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Conv2d(3, 4, 4)
        self.tconv = torch.nn.ConvTranspose3d(4, 1, 1)  
    def forward(self, x):
        v1 = F.avg_pool2d(F.relu(self.input(x)), kernel_size=4)
        return self.tconv(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
