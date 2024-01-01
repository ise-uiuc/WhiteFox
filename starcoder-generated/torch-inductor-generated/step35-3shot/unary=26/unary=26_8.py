
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(15, 7, kernel_size=(1, 5), stride=(1, 3), bias=False, padding=0)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 > 0
        v3 = v1 * 1.0
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x = torch.randn([1, 15, 40, 20])
