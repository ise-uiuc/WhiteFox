
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 19, 3, stride=2, padding=1), torch.nn.BatchNorm2d(19, momentum=0.8999999761581421, eps=1e-05), torch.nn.Tanh())
    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.ones((1, 3, 5, 10))
