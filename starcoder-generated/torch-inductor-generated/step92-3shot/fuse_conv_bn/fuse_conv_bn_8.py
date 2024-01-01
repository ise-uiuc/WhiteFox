
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.add_module("conv1", nn.Conv2d(1, 20, 5, 1))
        self.add_module("batchnorm1", nn.BatchNorm2d(20)),
        self.add_module("relu1", nn.ReLU())
        self.add_module("maxpool1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.add_module("bn2", nn.BatchNorm2d(20))
        self.add_module("conv2", nn.Conv2d(20, 20, 5, 1))
        model = [nn.ModuleDict(self.named_children())["conv1"], nn.ModuleDict(self.named_children())["batchnorm1"], nn.ModuleDict(self.named_children())["relu1"], nn.ModuleDict(self.named_children())["maxpool1"], nn.ModuleDict(self.named_children())["bn2"], nn.ModuleDict(self.named_children())["conv2"]]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.randn(2, 1, 28, 28)
