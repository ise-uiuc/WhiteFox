
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = 2
    def forward(self, x):
        y = x[:self.p]
        return x
# Inputs to the model
x = torch.randn(5, 3, 4)
