
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        x1 = torch.squeeze(x)
        x2 = torch.unsqueeze(x1, dim=1)
        return x2
