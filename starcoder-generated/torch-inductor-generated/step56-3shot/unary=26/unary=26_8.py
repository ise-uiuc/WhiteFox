
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: ConvTranspose2d 35 x 42 x 8 to 17 x 38 x 8
        self.conv_t_1 = torch.nn.ConvTranspose2d(42, 8, kernel_size=(8,8), stride=(2, 2))
        # Layer 2: ConvTranspose2d 17 x 38 x 8 to 4 x 233 x 16
        self.conv_t_2 = torch.nn.ConvTranspose2d(8, 64, (8, 16), padding=(3, 0))
        # Layer 3: ConvTranspose2d 4 x 233 x 16 to 1 x 369 x 32
        self.conv_t_3 = torch.nn.ConvTranspose2d(64, 369, (3, 32), padding=(1, 8), stride=(2, 2))
    def forward(self, x):
        