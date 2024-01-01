
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        t1 = x1.permute(0, 2, 1)
        t2 = self.linear(t1)
        return t2.transpose(1, 2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
