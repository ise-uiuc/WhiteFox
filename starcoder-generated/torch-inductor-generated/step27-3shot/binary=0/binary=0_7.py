
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 5, kernel_size=(1,1), stride=(1,1), bias=False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.sum(v1, dim=(2,3)).squeeze()
        return v2
# Inputs to the model
x = torch.randn(20, 4, 5, 5)
