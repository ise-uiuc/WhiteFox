
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.LSTM(16, 1024)
    def forward(self, x):
        return torch.tanh(self.features(x))
# Inputs to the model
x = torch.randn(16, 1, 1024)
