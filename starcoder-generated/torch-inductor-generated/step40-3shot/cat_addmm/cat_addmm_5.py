
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.layers_2 = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        s = x.mean(0)
        x = x.repeat_interleave(4, dim=0) # Repeat each element of x four times
        x = self.layers_2(x)
        x = torch.stack((x, x, x), dim=1)
        x = torch.flatten(x, 1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
