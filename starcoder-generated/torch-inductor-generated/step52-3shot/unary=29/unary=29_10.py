
class Model(torch.nn.Module):
    def __init__(self, min_value=-111.3, max_value=-80.3, padding=2, kernel_size=3):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 5, kernel_size, stride=2, padding=padding)
        self.act_3 = torch.nn.ReLU(inplace=True)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x7):
        v5 = self.conv_transpose2d(x7)
        v9 = self.min_value
        v12 = self.act_3(v5)
        v14 = self.max_value
        v16 = torch.clamp(v12, min=v9)
        v18 = torch.clamp(v16, max=v14)
        return v18
# Inputs to the model
x7 = torch.randn(1, 3, 13, 13)
