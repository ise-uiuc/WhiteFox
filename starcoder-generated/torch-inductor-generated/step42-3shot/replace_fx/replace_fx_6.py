
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ml = torch.nn.Linear(2, 2)
    def forward(self, x):
        x = self.ml(F.softmax(x))
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
