
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, Self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 2, 3, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
        self.avg_pool = torch.nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v3 = self.avg_pool(x1)
        v2 = torch.tanh(v1) + v3
        v4 = torch.relu(v2)
        v5 = torch.cat([v1, v2, v3, x1, v4], 1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
