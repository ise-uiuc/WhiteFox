
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = Layer1(3, 32, 64)
        self.extra = torch.nn.ModuleList([torch.nn.Softmax(dim=3)])
    def forward(self, v1):
        return self.extra[0](self.features(v1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
