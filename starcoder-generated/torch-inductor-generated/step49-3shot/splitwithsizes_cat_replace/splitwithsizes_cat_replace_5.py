
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(
            features=torch.nn.Sequential(
                torch.nn.Conv2d(3, 2, kernel_size=(3,)),
            )
        )
    def forward(self, value):
        r = self.features(value)
        return (r, torch.split(value, 2), r)
# Inputs to the model
x1 = torch.rand(1, 3, 32, 32)
