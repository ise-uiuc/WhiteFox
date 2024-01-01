
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 3, 3), nn.RelU(), nn.Conv2d(3, 3, 2, stride=(2,2), padding=1), nn.Conv2d(3, 3, 1, stride=(2,1), padding=2)
        )
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 2, 2)
