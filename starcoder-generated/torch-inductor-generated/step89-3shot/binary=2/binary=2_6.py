
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=False)
        self.conv2d_2 = torch.nn.Conv2d(4, 15, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
    def forward(self, x3):
        v1 = self.conv2d_1(x3)
        v2 = self.conv2d_2(v1)
        v3 = v2 - -8.319200057983398
        return v3
# Inputs to the model
x3 = torch.randn(1, 3, 64, 64)
