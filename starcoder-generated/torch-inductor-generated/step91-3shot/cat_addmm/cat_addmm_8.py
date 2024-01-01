
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 5)
        self.dropout = nn.Dropout()
        self.avg_pool1d = nn.AvgPool2d((3, 1))
        self.mean = torch.mean
    def forward(self, x):
        x = self.layers(x)
        x = self.dropout(x)
        x = x.unsqueeze(3)
        x = self.avg_pool1d(x)
        x = x.squeeze(3)
        x = self.mean(x, dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 3)
