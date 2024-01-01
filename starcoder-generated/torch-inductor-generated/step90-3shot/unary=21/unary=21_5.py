
class ModelDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 10, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 20, 3)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.9)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout1(x)
        return self.dropout2(x)
# Inputs to the model
input = torch.randn(1, 4, 16, 16)
