
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1).flatten(1)
        x1 = torch.cat([x, x], 0)
        x1 = x1.unsqueeze_(0)
        x2 = torch.cat([x, x], 1)
        x1, _, x2, _ = torch.chunk(
            torch.cat([x1, x2], 0), 4, dim=0)
        return x1, x2, torch.squeeze(x2, dim=0)
# Inputs to the model
x = torch.randn(2, 2)
