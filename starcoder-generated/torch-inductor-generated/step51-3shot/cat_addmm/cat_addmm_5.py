
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2*2, 1*2)
    def forward(self, x):
        x = self.layers(torch.cat((x, x), dim=0))
        x = x.view(1,2)
        return x
# Inputs to the model
x = torch.randn(2, 2)
