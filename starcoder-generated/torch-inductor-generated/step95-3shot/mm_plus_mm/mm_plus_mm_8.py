
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        #self.fc2 = nn.Linear(256, 128)
    def forward(self, x, x2):
        x = F.relu(self.fc1(x))
        x = self.fc1(x2)
        return x
# Inputs to the model
x = torch.randn(1, 256)
x2 = torch.randn(256, 256)
