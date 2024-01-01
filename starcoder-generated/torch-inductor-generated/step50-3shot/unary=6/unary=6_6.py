
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm3d = torch.nn.BatchNorm3d(
            num_features=36, eps=1e-05, momentum=0.1
        )
        self.conv = torch.nn.Conv3d(1, 1, 1)
    def forward(self, x1):
        t1 = self.batch_norm3d(x1)
        t2 = self.conv(t1)
        t3 = t1 + t2
        t4 = torch.clamp(t3, 0, 6)
        t5 = t3 - t4
        return t5
# Inputs to the model
x1 = torch.randn(2, 36, 128, 128, 64)
