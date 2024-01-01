
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(16, 16, 5, stride=2, padding=0, output_padding=0)
        self.conv1 = torch.nn.ConvTranspose2d(16, 16, 3, stride=2, padding=0, output_padding=0)
        self.convt = torch.nn.MaxUnpool2d(2)
    def forward(self, x1):
        v2 = self.conv(x1)
        v3 = torch.relu(v2)
        v4 = F.max_unpool2d(v3, self.argmax_v3, 2, 0, 0)
        v5 = self.conv1(v4)
        v6 = torch.relu(v5)
        v7 = F.unfold(v6, 2, 1, 0)
        v8 = self.convt(v7, self.argmax_v8)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 8, 8)
