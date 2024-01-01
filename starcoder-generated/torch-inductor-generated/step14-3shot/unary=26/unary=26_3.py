
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(192, 128, (1, 1), stride=(1, 1))
        self.batch_norm = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU()
    def forward(self, x0):
        identity = x0
        v0 = self.conv_t(x0)
        v1 = self.batch_norm(v0)
        v2 = self.relu(v1)
        v3 = torch.add(identity, v2)
        return v3
# Inputs to the model
x0 = torch.randn(1, 192, 7, 7)
