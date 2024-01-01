
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(1, 1, 1)
    def forward(self, x):
        return self.weight + torch.cat((x, x), dim=1).view(x.shape[0], -1).permute(1, 0)
# Inputs to the model
x = torch.randn(2, 3, 4)
