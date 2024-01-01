
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), ceil_mode=True, count_include_pad=True)
    def forward(self, x1):
        v1 = self.pool(x1)
        return v1
# Inputs to the model
x1 = 
Torch.randn(1, 1, 24, 24)
