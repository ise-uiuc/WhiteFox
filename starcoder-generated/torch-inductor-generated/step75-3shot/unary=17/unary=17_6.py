
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 12, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(12, 1, 3, stride=1, padding=1),
            #nn.Sigmoid()
        )
    def forward(self, x1):
        #print(x1.shape)
        output = self.model(x1)
        #print(output.shape)
        return output
# Inputs to the model
x1 = torch.randn(1, 1, 300, 300)
