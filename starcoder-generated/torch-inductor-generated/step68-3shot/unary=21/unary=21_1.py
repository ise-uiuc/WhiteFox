
class ModelTanhWithReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        h1 = self.relu(x)
        h2 = self.conv(h1)
        h3 = torch.tanh(h2)
        return h3
# Inputs to the model
x = torch.rand(1, 3, 64, 64)
