
class Model(torch.nn.Module):
    def __init__(self, device='cpu', min_value=-4.8302, max_value=6.9388):
        super().__init__()
        self.pad = torch.nn.ReplicationPad2d(2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.pad(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 5, 4)
