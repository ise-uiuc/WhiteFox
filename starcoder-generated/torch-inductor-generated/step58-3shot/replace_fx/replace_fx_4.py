
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, 3, padding = 1)
        self.conv2 = torch.nn.Conv2d(128, 64, 3, padding = 1)
    def forward(self, x):
        x1 = F.dropout(self.conv1(x), p=0.5, training=self.conv1.training)  # Conv1 is training
        x2 = F.dropout(self.conv1(x), p=0.5, training=False)  # Conv1 is not training
        x3 = self.conv2(x1)
        x4 = x3 + F.dropout(x3, p=0.5, training=self.conv1.training)
        x5 = F.dropout(self.conv2(x2), p=0.5, training=self.conv2.training)  # Conv2 is training
        x6 = F.dropout(self.conv2(x4), p=0.5, training=self.conv2.training)  # Conv1 and Conv2 are training
        return x6
# Inputs to the model
x = torch.randn(20, 64, 50, 20)
