
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(2, 3, 20, 1, 15)
    def forward(self, img):
        x = F.relu(self.conv_t(img))
        return torch.where(x < 0., x, torch.zeros_like(x))
# Inputs to the model
img = torch.randn(1, 2, 50, 35)
