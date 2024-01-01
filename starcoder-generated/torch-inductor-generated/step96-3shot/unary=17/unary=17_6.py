
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 20, kernel_size=5, padding=0)
        self.fc1 = torch.nn.Linear(400, 120)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        h1 = v2.view(-1, 400)
        v3 = self.fc1(h1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
