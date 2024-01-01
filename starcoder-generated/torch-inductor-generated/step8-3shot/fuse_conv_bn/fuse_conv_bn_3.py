
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        conv = nn.Conv2d(2, 3, (5, 5))
        relu = nn.ReLU()
        bn = nn.BatchNorm2d(3)
        cat = torch.cat
        seq = nn.Sequential(
            conv,
            relu,
            bn,
            cat
        )
        self.model = seq
    def forward(self, inputs):
        return self.model(inputs)
## Inputs to the model
batch = 2
input = torch.randn((batch, 2, 5, 5), dtype=torch.float)
#