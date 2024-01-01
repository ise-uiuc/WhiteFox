
class ModelNew1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 2)
        self.dropout = torch.nn.Dropout(0.1)
        self.conv2 = torch.nn.Conv2d(6, 2, 2)
        self.linear1 = torch.nn.Linear(2, 3)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = torch.rand_like(x)
        x = torch.nn.functional.dropout(x)
        x = self.linear1(x)
        return x

# Inputs to the model
x1 = torch.randn(2, 3, 2, 2)
