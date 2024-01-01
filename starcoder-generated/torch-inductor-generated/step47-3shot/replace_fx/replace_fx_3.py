
class model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.dropout = torch.nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv2(x)
        x = torch.nn.functional.sigmoid(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x1 = torch.rand(1, 1, 28, 28) * 255
