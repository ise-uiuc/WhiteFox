
class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.w = torch.randn(5)
    def forward(self, x):
        x = x + self.w
        return x
# Inputs to the model
x = torch.randn(5)
