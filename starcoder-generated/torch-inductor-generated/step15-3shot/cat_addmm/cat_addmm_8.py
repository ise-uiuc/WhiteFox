
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.layers(x)
        x = self.relu(x)
        x = torch.cat((x, x, x), dim=1)
        return x
