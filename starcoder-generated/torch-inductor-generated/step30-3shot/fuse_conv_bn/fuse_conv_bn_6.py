
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv1d(2, 2, 2)
        self.fc = torch.nn.Linear(2, 2)
        self.bn = torch.nn.BatchNorm1d(2)
    def forward(self, input_1, input_2):
        x = self.conv(input_1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.softmax(x)
        x = self.relu(x)
        x = x + input_2
        return x
# Inputs to the model
input_1 = torch.randn(1, 2, 8)
input_2 = torch.randn(1, 2, 8)
