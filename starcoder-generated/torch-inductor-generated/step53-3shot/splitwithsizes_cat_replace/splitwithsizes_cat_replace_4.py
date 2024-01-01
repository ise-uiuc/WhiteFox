
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, bias=True)
    def forward(self, input):
        return (torch.cat(torch.split(input, [1, 1, 1], dim=2), dim=2), torch.split(input, [1, 1, 1], dim=2))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
