
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 16, (5, 5), 1, (2, 2))
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.4)
    def forward(self, input):
        x = self.conv(input)
        x = self.dropout1(x)
        x = self.dropout2(x)
        x = torch.rand_like(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 88, 88, 2)
