
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x1):
        x2 = self.linear1(x1)
        x2 = self.relu(x2)
        x3 = self.linear2(x2)
        x4 = torch.nn.functional.dropout(x3)
        x5 = F.dropout(x4)
        x6 = self.relu(x5)
        x7 = x1 * x6
        return x7

class Nested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(10, 10)
        self.convblock2 = ConvBlock(10, 10)

    def forward(self, x1):
        x7 = self.convblock1(x1)
        x8 = self.convblock2(x7)
        return x8

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nested = Nested()

    def forward(self, x1):
        x9 = self.nested(x1)
        x10 = torch.nn.functional.dropout(x9)
        return x10

# Inputs to the model
x1 = torch.rand(3, 10)
