
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x), dim=1)
        x = torch.split(x, split_size_or_sections=int(x.shape[1]/2), dim=1)
        x = torch.stack(x, dim=2)
        x = torch.sum(x, dim=3)
        return x
# Inputs to the model
x = torch.randn(2, 8)
