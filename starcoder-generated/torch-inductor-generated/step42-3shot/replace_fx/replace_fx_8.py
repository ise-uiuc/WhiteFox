
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = nn.Linear(2, 2)
    def forward(self, x):
        x1 = torch.rand(1, 2)
        x2 = torch.rand(2, 2)
        x = self.sub(x).sum() + x1 + torch.einsum("abc,cb->ac", [x2, x2])
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
