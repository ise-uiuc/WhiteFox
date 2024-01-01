
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(3, 8, kernel_size=3), nn.ReLU(), nn.AdaptiveAvgPool2d(output_size=(5, 5)), nn.Dropout(),
                           nn.Linear(168*8*8, 10), nn.Softmax())] * 16
            )
    def forward(self, x):
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in {0, 4, 9, 13}:
                outputs.append(x)
        return tuple(outputs)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
