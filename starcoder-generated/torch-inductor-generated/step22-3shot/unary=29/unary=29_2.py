
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super(Model, self).__init__()
        self.add = torch.nn.Add()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        x1 = torch.randn(1, 64, 128, 128)
        x2 = torch.randn(1, 64, 128, 128)
        x = self.add(x, x1, x2)
        return x.clamp(self.min_value, self.max_value)
min_value = -0.5
max_value = 5.4
# Inputs to the model
x = torch.randn(1, 128, 224, 224)
