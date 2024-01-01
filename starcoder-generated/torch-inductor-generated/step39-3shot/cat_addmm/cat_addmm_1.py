
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 8)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x, x, x, x, x), dim=1)
        x = torch.cat((x[:, 3:].unsqueeze(0).repeat_interleave(2, dim=0), x[:, 0:3].unsqueeze(0).repeat_interleave(2, dim=0)), dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
