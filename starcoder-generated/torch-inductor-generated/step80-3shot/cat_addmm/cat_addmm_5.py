
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 4)
        self.layers2 = nn.Conv2d(1, 1, kernel_size=2, stride=1)
    def forward(self, x):
        x = self.layers(x)
        x = self.layers2(x.view(1, 1, 2, 2))
        x = torch.stack([x[..., 0], x[..., 1]], dim=1)
        x = x.squeeze(1)
        x = torch.nn.functional.instance_norm(x, affine=True)
        x = x[..., None].squeeze(-1)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2, 2)
