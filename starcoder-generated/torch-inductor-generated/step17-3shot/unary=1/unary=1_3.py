
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg11_bn()
 
        # This section of code selects the layer in the PyTorch model graph associated with the output 'features.33' and renames it 'conv'.
        t = list(self.vgg.features.children())
        self.vgg.features = torch.nn.Sequential(
            *t[:16],
            self.vgg.features.conv3)
 
    def forward(self, x1):
        m1 = self.vgg(x1)
        return m1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
