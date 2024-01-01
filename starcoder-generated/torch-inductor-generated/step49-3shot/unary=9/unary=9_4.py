
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = torch.floor(x1) + 3
        v2 = torch.clamp_min(v1, 0.0)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, 128, 128, requires_grad=True)
