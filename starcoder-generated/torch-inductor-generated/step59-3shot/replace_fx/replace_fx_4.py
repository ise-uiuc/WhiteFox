
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.ones(2, 2, 2)
    def forward(self, x):
        x = torch.where(x > self.t1[0], x + self.t1[0], self.t1[1])
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
