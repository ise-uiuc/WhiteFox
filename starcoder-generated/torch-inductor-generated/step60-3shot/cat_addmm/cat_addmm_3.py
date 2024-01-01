
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(512, 1024)
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return x.unsqueeze_(1)
# Inputs to the model
x = torch.randn(2, 512)
