
class Model(nn.Module):
    def __init__(self, dropout_rate=0.9):
        super().__init__()
        self.blocks = []
        layer1 = list([
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate/2)
        ])
        layer2 = list([
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate/2)
        ])
        self.blocks.extend([
            nn.Sequential(*layer1),
            nn.Sequential(*layer2),
            nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_rate),
            )
        ])
    
    def forward(self, x):
        for i in range(3):
            x = self.blocks[i](x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 28, 28)
