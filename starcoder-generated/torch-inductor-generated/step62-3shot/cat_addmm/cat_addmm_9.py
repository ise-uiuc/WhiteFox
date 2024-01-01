
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 6)
    def forward(self, x):
        x = self.layers(x)
        x = x[1][::4] # 4x4 pixel sampling pattern
        x = x.reshape(6, 4) # 6 x 4 is the new shape
        return x
# Inputs to the model
x = torch.randn(1, 4, 4, 4)
