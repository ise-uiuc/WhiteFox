
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4, bias=False)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=1)
        x = torch.flatten(x, 1)
        x = x.unsqueeze_(1)
        x = torch.cat((x.unsqueeze(0), x.unsqueeze(0)), dim=0)
        x = x.repeat_interleave(3, dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
model = Model()
model.eval()
