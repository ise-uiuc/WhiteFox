
class Model(object):
    def __init__(self) -> None:
        self.layer1 = torch.nn.Linear(2, 2)
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = self.layer1(x)
        x = x.view(x.shape[0], -1)
        x = x.tanh()
        x = x.view(x.shape[0], -1)
        x = x.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
