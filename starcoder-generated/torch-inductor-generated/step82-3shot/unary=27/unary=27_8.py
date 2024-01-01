
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.depth_to_space1d = torch.nn.functional.depth_to_space(block_size=2, spatial_dim=3)
        self.linear = torch.nn.Linear(in_features=12, out_features=14, bias=False)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.depth_to_space1d(x1)
        v2 = self.linear(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.6
max = 0.7
# Inputs to the model
x1 = torch.randn(5, 12, 2)
x2 = torch.randn(3, 10)
