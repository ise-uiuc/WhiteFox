
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.conv(x1).add(3)
        v2 = v1.clamp(min=0, max=6)
        v3 = v2.div(6)
        return v3


    model = Model()
    input = {"x1": torch.randn(1, 3, 64, 64)}
    model(**input)

    return model, input
# Input to the model
model, input = create()
