
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._21 = nn.ConvTranspose2d(5, 4, (3, 3), stride=1, padding=(1, 1), bias=True)

    def forward(self, x1):
        v1 = self._21(x1)
        v2 = torch.clamp_min(v1, -0.662077819824219)
        v3 = torch.clamp_min(v2, -0.0572269584655761)
        v4 = torch.clamp_max(v3, 0.2742073059082031)
        v5 = v4.detach()
        return v5
# Inputs to the model
x1 = torch.randn(1, 5, 3, 2)
