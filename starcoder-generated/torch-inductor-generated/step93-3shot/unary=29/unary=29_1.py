
class Model(torch.nn.Module):
    def __init__(self, min_value=-433.12, max_value=-270.08):
        super().__init__()
        self.batch_norm = torch.nn.ModuleList([torch.nn.Linear(7, 32, bias=True), torch.nn.BatchNorm2d(32)])
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.batch_norm[0].conv1d(x1)
        v2 = self.batch_norm[1](v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 7, 5, 5)
