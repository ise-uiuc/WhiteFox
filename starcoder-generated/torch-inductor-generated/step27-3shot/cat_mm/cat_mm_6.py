
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = torch.randn_like(x2).transpose(-1, -2).contiguous()
        v1 = torch.nn.functional.interpolate(x1, mode='nearest', scale_factor=2)*(1.0/torch.sqrt(2))
        v2 = torch.nn.functional.interpolate(x1, mode='nearest', scale_factor=0.1)*(1.0/torch.sqrt(2))
        return torch.cat([v1, v1, v2, v2], 2).contiguous()
# Inputs to the model
x1 = torch.randn(2, 2, 2)
x2 = torch.randn(2, 2, 2)
