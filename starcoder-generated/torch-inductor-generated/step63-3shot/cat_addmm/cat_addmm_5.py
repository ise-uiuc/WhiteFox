
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        self.layers2 = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = x.unsqueeze(dim=2)
        x = torch.transpose(x, 1, 2)
        x = x.flatten(start_dim=2, end_dim=3)
        return x
# Inputs to the model
x = torch.randn(2, 2)
