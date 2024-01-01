
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(256, 512, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv1d(512, 2051, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv1(x2)
        v1 = v1.permute(0,2,1)
        v1 = v1.flatten(start_dim=2)
        v2 = torch.cat([x1, v1], dim=1)
        v3 = self.conv2(v2)
        v3 = v3.permute(0,2,1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 2)
x2 = torch.randn(1, 128, 112)
