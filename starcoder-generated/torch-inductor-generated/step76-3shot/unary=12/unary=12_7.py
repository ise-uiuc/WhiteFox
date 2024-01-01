
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential()
        self.conv_0 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        self.conv_1 = nn.ModuleList()
        for i in range(2):
            self.conv_1.add_module(("conv" + str(i), nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)))
        self.conv_2 = nn.ModuleDict({'conv_2': nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0), 'conv_3': nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)})
        self.conv_4 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = F.relu(self.conv_0(x1))
        for i, conv in enumerate(self.conv_1):
            v1 = F.relu(conv(v1))
        v1 = self.conv_2['conv_2'](v1) + self.conv_2['conv_3'](v1)
        v1 = F.relu(self.conv_4(v1))
        v2 = v1 * torch.sigmoid(v1)
        v3 = torch.sigmoid(torch.exp(-v2))
        v3 = torch.tanh(v3)
        v2 = v3 * v2
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
