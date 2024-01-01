
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x):
        y = self.linear(x).view(x.shape[0], -1)
        return y.tanh() if x.shape!= (1, 3) else y.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
