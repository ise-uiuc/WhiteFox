
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1 + 3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(512, 512)
