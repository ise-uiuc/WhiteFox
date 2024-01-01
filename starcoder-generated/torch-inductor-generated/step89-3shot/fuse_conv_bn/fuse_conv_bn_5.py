
class Model(nn.Module):
    def __init__(self):
       super().__init__()
       self.num_classes = 10 
       self.features = self._make_layers() 
       self.classifier = nn.Linear(512, self.num_classes)
       self.conv4 = nn.Conv2d(32,64,(1,1),1,bias=False)
       self.conv5 = nn.Conv2d(32,64,(1,1),1,bias=False)
       self.bn4 = nn.BatchNorm2d(64)
       self.bn5 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = self.features(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
       layers = []
       in_channels = 176
       #conv0
       layers+=[nn.Conv2d(176, 32, (3,3), 1, 1)]
       #stage 1
       layers+=self._make_stage(in_channels, 32, num=10, with_pooling=True)
       #stage 2
       in_channels = 192
       layers+=self._make_stage(in_channels, 64,num=13, with_pooling=True)
       #stage 3
       in_channels = 128
       layers+=self._make_stage(in_channels,64, 32,num=6, with_pooling=False)
       #stage 4
       in_channels = 256
       layers+=self._make_stage(in_channels,64, 32,num=6, with_pooling=False)
       #stage 5
       in_channels = 384
       layers+=self._make_stage(in_channels,64, 32,num=6, with_pooling=False)
       return nn.Sequential(*layers)

    def _make_stage(self, in_channels, out_channels_1, out_channels_2=0, group = 32, num=1, with_pooling = False):
       layers = []
       #conv3x3
       layers+=[ nn.BatchNorm2d(in_channels), nn.ReLU(), nn.Conv2d(in_channels, out_channels_1, (3,3), 1, 1, bias=False), nn.BatchNorm2d(out_channels_1), nn.ReLU()]
       for i in range(num-1):
           layers += [nn.Conv2d(in_channels, out_channels_1, (3,3), 1, 1, bias=False), nn.BatchNorm2d(out_channels_1), nn.ReLU(), nn.Conv2d(out_channels_1, out_channels_1, (3,3), 1, 1, bias=False), nn.BatchNorm2d(out_channels_1), nn.ReLU()]
       if out_channels_2!=0:
            layers+=[nn.Conv2d(out_channels_1, out_channels_2, (1,1))]
            if with_pooling:
                layers.append(nn.AvgPool2d((7, 7), 1))
       return layers
# Inputs to the model
x = torch.randn(1, 176, 4, 4)
