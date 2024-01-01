
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv3d(3, 3, 3)
        torch.manual_seed(2)
        self.dropout = torch.nn.Dropout2d(p=0.3, inplace=True)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(3, 3, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 10, 10, 10)
