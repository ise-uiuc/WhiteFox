
class Block(torch.nn.Module):
    def forward(self, x1):
        x2 = torch.mm(x1, x1)
        x3 = torch.cat([x2, x2], 1)
        return x3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def forward(self, x1, x2):
        return self.block(x1)
# Input to the model
x1 = torch.randn(16, 32)
x2 = torch.randn(32, 8)
