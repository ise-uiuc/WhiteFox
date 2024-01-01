
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(10, 10)
    def forward(self, x):
        x1 = x
        x1 = self.layers(x1)
        x1 = torch.stack([x1, x1, x1], dim=0)

        def fun(x2):
            return (x2.shape[0])

        x2 = x1.apply(fun)
        x2 = torch.stack([x2, x2])
        return x2
# Inputs to the model
x = torch.randn(20, 10)
