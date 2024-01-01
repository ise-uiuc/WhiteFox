
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(512, 256, 1)
        self.linear = torch.nn.Linear(256, 512)
        self.tanh = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv1d(512, 5, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.conv1(x)
        x = self.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 512)
