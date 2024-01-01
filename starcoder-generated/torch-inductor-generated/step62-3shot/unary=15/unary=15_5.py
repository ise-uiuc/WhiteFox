
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = torchvision.models.vgg16()
        self.vgg16.classifier = torch.nn.Sequential()
        self.vgg16.avgpool = torch.nn.Sequential()
        self.dropout1 = torch.nn.Dropout(0.5)
        self.last_fc = torch.nn.Linear(1000, 10)
    def forward(self, x1):
        v1 = self.vgg16(x1)

        v2 = self.dropout1(v1)
        v3 = self.last_fc(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 224, 224)
