
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model = nn.Sequential(nn.Conv2d(3,10,1), nn.Conv2d(10,10,3), nn.MaxPool2d(2))
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.rand(2, 3, 256, 256)
