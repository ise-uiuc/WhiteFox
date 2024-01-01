
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_filters, out_filters, kernel_size=3, stride=1):
            layers = [nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride, padding=1), nn.BatchNorm2d(out_filters), nn.ReLU(True)]
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            
            block(2, 16),
            block(16, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
            
        )

    def forward(self, z):
        return self.model(z)
# Inputs to the model
z = torch.randn(1, 2)
