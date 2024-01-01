
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(1, 2, 4, stride=2, padding=0, output_padding=4)
        self.relu = torch.nn.BatchNorm1d(num_features=2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    def forward(self, x47):
        v1 = self.conv(x47)
        v2 = v1.permute(0, 2, 1)
        v3 = self.relu(v2)
        v4 = v3.permute(0, 2, 1)
        return v4
# Inputs to the model
x47 = torch.randn(1, 1, 17)
