
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            F.conv1d(16, 33, 3, stride=2),
            nn.BatchNorm1d(33),
            nn.ReLU(),
            F.conv1d(33, 67, 4),
            nn.BatchNorm1d(67),
            nn.ReLU(),
            F.conv1d(67, 9, 5),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            F.adaptive_max_pool1d(4, return_indices=True)
        )
    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.randn(2, 16, 20)
