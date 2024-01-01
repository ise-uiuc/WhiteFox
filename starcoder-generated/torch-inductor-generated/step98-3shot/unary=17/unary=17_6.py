
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(64, 3, 3, 1, padding='valid')
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x
    
model = Model()
# Inputs to the model
x = torch.randn(1, 3, 224, 244)
