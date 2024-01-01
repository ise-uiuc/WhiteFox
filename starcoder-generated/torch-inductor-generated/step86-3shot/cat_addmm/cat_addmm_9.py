
class M(nn.Module):
    def __init__(self, n_classes=345):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                nn.ConvTranspose1d(256, 512, kernel_size=(10,), stride=(2,), padding=5),
                nn.ConvTranspose1d(512, 512, kernel_size=(10,), stride=(1,), padding=5),
                nn.ConvTranspose1d(512, 512, kernel_size=(10,), stride=(1,), padding=5),
                nn.ConvTranspose1d(512, n_classes, kernel_size=(10,), stride=(2,), padding=5),
            )
        )

    def forward(self, x):
        for l in self.layers:
            x = l(x)
            x = nn.functional.gelu(x)
        return x
# Inputs to the model
x = torch.randn(2, 256, 10)
