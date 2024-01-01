
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Conv2d(16, 16, 1, padding=0),
             torch.nn.Conv2d(16, 16, 3, padding=1),
             torch.nn.Conv2d(16, 16, 5, padding=2)])
    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x
            x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
