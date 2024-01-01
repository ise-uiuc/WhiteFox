
class Model(torch.nn.Module):
    def __init__(self, min_value=-126.73123535322572, max_value=127.36052785355157):
        super().__init__()
        self.elu = torch.nn.ELU(alpha=-14.641941431417867, inplace=False)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, bias=True, padding=15)
        self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.avgpool(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        v5 = self.elu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
