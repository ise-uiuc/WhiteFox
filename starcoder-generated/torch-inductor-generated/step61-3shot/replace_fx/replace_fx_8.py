
class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('d1', nn.Dropout())
        self.model.add_module('d2', nn.Dropout())
        for param in self.model.parameters():
            nn.init.constant_(param, 1)

    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.randn(2)
