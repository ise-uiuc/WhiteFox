
class Model(torch.nn.Module):
    def __init__(self, input_ch, num_classes, **kwargs):
        super(Model, self).__init__()
        self.input_ch = input_ch
        self.num_classes = num_classes
        self.features = self._make_layers(**kwargs)
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _make_layers(self):
        layers = []
        in_channels = 1
        layers += [conv_3x3_bn(self.input_ch, 64, stride=2)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)
# Inputs to the model
x = torch.randn(1, 1, 224, 224)
