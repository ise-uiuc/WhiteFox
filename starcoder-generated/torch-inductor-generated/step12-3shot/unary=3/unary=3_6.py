
class Module1560(torch.nn.Module):
    def __init__(self, conv1_in_channels, conv1_out_channels, conv1_kernel_size, conv1_stride, conv1_padding,
                 conv2_in_channels, conv2_out_channels, conv2_kernel_size, conv2_stride, conv2_padding):
        super(Module1560, self).__init__()
        self.conv1 = torch.nn.Conv2d(conv1_in_channels, conv1_out_channels, conv1_kernel_size, stride=conv1_stride,
                                 padding=conv1_padding, bias=False)
        self.conv2 = torch.nn.Conv2d(conv2_in_channels, conv2_out_channels, conv2_kernel_size, stride=conv2_stride,
                                 padding=conv2_padding, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(conv1_out_channels, eps=9.99999974738e-06, momentum=0.0,
                                 affine=True, track_running_stats=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out

class Module1561(torch.nn.Module):
    def __init__(self):
        super(Module1561, self).__init__()
        self.conv1 = torch.nn.Conv2d(195, 32, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(32, eps=9.99999974738e-06, momentum=0.0, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(32, 128, 1, stride=1, padding=0)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv_block(out)
        return out

class Module1562(torch.nn.Module):
    def __init__(self):
        super(Module1562, self).__init__()
        self.conv1 = torch.nn.Conv2d(128, 32, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(32, eps=9.99999974738e-06, momentum=0.0, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(32, 704, 1, stride=1, padding=0)
        self.flatten = torch.nn.Flatten(1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.flatten(out)
        return out

class Module1563(torch.nn.Module):
    def __init__(self, class_num_list, use_aux, init_bias, use_dropout, use_bn, use_gn, num_layers):
        super(Module1563, self).__init__()
        layers = [Module1574(in_channels=588, mid_channels=128, out_channels=256, num_layers=num_layers)]
        if use_aux:
            layers.append(Module1564(out_channels=512, num_classes=class_num_list[1]))
        layers.append(Module1570(in_channels=1792, out_channels=768, num_layers=num_layers))
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = ClassifierModule(768, class_num_list, use_aux, init_bias, use_dropout, use_bn, use_gn)
        self.flatten = torch.nn.Flatten(1)
    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.classifier(out)
        out = self.flatten(out)
        return out

class Module1564(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Module1564, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=128, eps=9.99999974738e-06, momentum=0.0, affine=True, track_running_stats=True)
        self.flatten = torch.nn.Flatten(1)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.linear = torch.nn.Linear(in_features=13504, out_features=num_classes, bias=True)
        self.softmax = torch.nn.Softmax(-1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.flatten(out)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.softmax(out)
        return out

class Module1565(torch.nn.Module):
    def __init__(self):
        super(Module1565, self).__init__()
        self.conv_block1 = Module1560(conv1_in_channels=512, conv1_out_channels=288, conv1_kernel_size=(1, 1), conv1_stride=(1, 1), conv1_padding=(0, 0),
                conv2_in_channels=288, conv2_out_channels=768, conv2_kernel_size=(3, 3), conv2_stride=(2, 2), conv2_padding=(1, 1))
        self.conv_block2 = Module1560(conv1_in_channels=768, conv1_out_channels=512, conv1_kernel_size=(1, 1), conv1_stride=(1, 1), conv1_padding=(0, 0),
                conv2_in_channels=512, conv2_out_channels=1536, conv2_kernel_size=(3, 3), conv2_stride=(1, 1), conv2_padding=(1, 1))
        self.conv = torch.nn.Conv2d(in_channels=1536, out_channels=195, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv(out)
        return out

class Module1566(torch.nn.Module):
    def __init__(self):
        super(Module1566, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=896, out_channels=195, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
    def forward(self, x):
        out = self.conv(x)
        return out

class Module1567(torch.nn.Module):
    def __init__(self):
        super(Module1567, self).__init__()
        self.module1565_0 = Module1565()
        self.module1566_1 = Module1566()
        self.concat = torch.cat([1], [1])
    def forward(self, x):
        out = [self.module1565_0(x), self.module1566_1(x)]
        out = self.concat(out, 1)
        return out

class Module1568(torch.nn.Module):
    def __init__(self):
        super(Module1568, self).__init__()
        self.concat = torch.cat([1], [1])
    def forward(self, x):
        out = [x, x]
        out = self.concat(out, 1)
        return out

class Module1569(torch.nn.Module):
    def __init__(self, out_channels, num_layers):
        super(Module1569, self).__init__()
        self.layers = ModuleList()
        self.layers.add_module(str(len(self.layers)), Module1571(in_channels=out_channels, mid_channels=out_channels, out_channels=out_channels, num_layers=num_layers))
        self.layers.add_module(str(len(self.layers)), Module1566())
    def forward(self, x):
        for layer in self.layers.children():
            x = layer(x)
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet34 = torchvision.models.resnet34(pretrained=True)
        # (Pdb) resnet34.features.children()
        # ModuleList(
        #   (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (2): ReLU(inplace=True)
        #   (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        #   (4): Sequential(
        #     (0): BasicBlock(
        #       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #     (1): BasicBlock(
        #       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #   )
        #   (5): Sequential(
        #     (0): BasicBlock(
        #       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (downsample): Sequential(
        #         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        #         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       )
        #     )
        #     (1): BasicBlock(
        #       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     )
        #   )
        #   (6): Sequential(
        #     (0): BasicBlock(
        #       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (downsample): Sequential(
        #         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        #         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       )
        #     )
        #     (1): BasicBlock(
        #       (conv1): Conv2d(256, 256, kernel_size=(3, 