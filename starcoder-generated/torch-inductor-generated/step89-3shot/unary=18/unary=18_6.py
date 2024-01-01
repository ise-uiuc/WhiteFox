
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 6, kernel_size=1, stride=1, padding='valid')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = F.adaptive_max_pool1d(v2, 1)
        v4 = F.interpolate(v3, size=[16])
        v5 = torch.cat([v3,v2,v4],dim=1)
        v6 = F.max_pool1d(nn.functional.gelu(v5), v5.size()[2:])
        v7 = self.conv(v6)
        v8 = F.softmax(v7, dim=1)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 32)
# Expected output of the model
output1 = torch.sigmoid(x1)
