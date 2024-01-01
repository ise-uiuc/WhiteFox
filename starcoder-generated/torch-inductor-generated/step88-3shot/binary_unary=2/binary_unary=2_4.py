
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.BatchNorm1d(8, affine=False, track_running_stats=False)
        self.norm2 = torch.nn.BatchNorm1d(8, affine=False, track_running_stats=False)
        self.norm3 = torch.nn.BatchNorm1d(8, affine=False, track_running_stats=False)
        self.norm4 = torch.nn.BatchNorm1d(8, affine=False, track_running_stats=False)
    def forward(self, x1):
        v1 = self.norm1(x1)
        v2 = self.norm2(v1)
        v3 = self.norm3(v2)
        v4 = self.norm4(v3)
        return v4

