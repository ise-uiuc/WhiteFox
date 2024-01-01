
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, inp):
        a = torch.split(
            inp[0],
            (32, 16, 32, 16),
            dim=1)
        concat_a = torch.cat([a[x] for x in range(len(a))], dim=1)
        return [concat_a]

# Initializing the model
m = Model()


# Inputs to the model
x1 = torch.randn(1, 23, 64, 64)
