
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loopVar1 = torch.mm(torch.ones(1, 1), torch.ones(1, 1))
    def forward(self, x1):
        for loopVar1 in range(5):
            self.loopVar1 = torch.mm(torch.ones(1, 1), torch.ones(1, 1))
        a = torch.mm(torch.ones(1, 1), torch.ones(1, 1))
        return torch.cat([self.loopVar1, a], 1)
# Inputs to the model
x1 = torch.randn(5, 5)
