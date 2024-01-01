
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.BatchNorm2d(1), torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1)))
    def forward(self, x1):
        v1 = self.convs(x1)
        v2 = torch.conv1d(v1,v1,v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 112)
