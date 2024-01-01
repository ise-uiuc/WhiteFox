
class Model(torch.nn.Module):
    def __init__(self):
        super(torchvision.models.mobilenet.MobileNet, self).__init__()
        self.model = torchvision.models.mobilenet.MobileNet()
        self.block = torch.nn.Sequential(self.model.features, self.model.conv, self.model.avgpool)
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
