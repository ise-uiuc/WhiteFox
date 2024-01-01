
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 6, bias=False)
        self.linear2 = nn.Linear(6, 7)
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
# Inputs to the model
x = torch.randn(2, 5)
