
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, [3, 3], stride=1, padding=1, bias=False),
            torch.nn.Conv2d(128, 256, [1, 1], stride=2, padding=0, bias=False)
        )
 
    def forward(self, images):
        x = self.features(images)
        x = x.view(x.size(0), -1)
        output = x - images
        return output

# Initializing the model
m = Model()

# Inputs to the model
__x__ = torch.randn(1, 3, 64, 64)
