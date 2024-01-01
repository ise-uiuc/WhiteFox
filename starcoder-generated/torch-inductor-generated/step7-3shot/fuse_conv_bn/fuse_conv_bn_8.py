
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        bn1.weight.d.requires_grad = False
        # TODO: use add_module to construct the model, self.features1 is a ModuleDict in PyTorch
        self.features1 = nn.ModuleDict()
        self.features1.bn1 = bn1
        self.features1.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.features1.relu1 = nn.ReLU(inplace=False)
        self.features1.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.features1.bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.features1.relu2 = nn.ReLU(inplace=False)
    def forward(self, x):
        x = self.features1['bn1'](x)
        x = self.features1['relu1'](x)
        x = self.features1['conv1'](x)
        x = self.features1['bn2'](x)
        x = self.features1['relu2'](x)
        x = self.features1['conv2'](x)
        return x
# Inputs to the model
x = torch.randn(2, 256, 56, 56) 
