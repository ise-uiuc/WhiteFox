
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(nn.Conv1d(1, 1, kernel_size=1), nn.BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True))
    def forward(self, x):
        return self.layers(x)
# Inputs to the model
x = torch.randn(4, 1, 1)
