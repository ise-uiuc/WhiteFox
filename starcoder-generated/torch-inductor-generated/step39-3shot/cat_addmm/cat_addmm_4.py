
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(60, 50)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=1)
        x = x.squeeze(0)
        x = torch.cat((x.unsqueeze(0), x.unsqueeze(0)), dim=0)
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.repeat_interleave(x, 3, dim=0)
        x = x + 5
        x = x.flatten()
        return x
# Inputs to the model
x = torch.randn(20, 60)
