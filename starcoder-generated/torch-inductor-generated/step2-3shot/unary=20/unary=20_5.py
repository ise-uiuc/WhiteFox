
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c2s = nn.ConvTranspose2d(4, 8, (3, 3), stride=(2, 2))
        self.h2h = nn.ConvTranspose2d(4, 8, (3, 3), (1, 1), (1,1), (1, 1))
        self.g2g = nn.ConvTranspose2d(4, 8, (3, 3), (1, 1), (1,1), (1, 1))
    def forward(self, x0):
        x1 = F.relu(self.c2s(x0)) + F.relu(self.g2g(x0)) + (self.h2h(x0))
        x2 = torch.sigmoid(x1)
        return x2
# Inputs to the model
x0 = torch.tensor(np.ones(shape=(1, 4, 32, 32), dtype=np.float32))
