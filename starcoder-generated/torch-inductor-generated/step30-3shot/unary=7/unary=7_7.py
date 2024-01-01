
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_features)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(torch.min(v1), min=0.0, max=6.0)
        v3 = v2 / 6
        return v3

