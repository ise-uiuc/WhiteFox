
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(input_size=32, hidden_size=32, num_layers=1, bias=True, batch_first=False, dropout=0.1, bidirectional=False)
        self.flatten1 = torch.nn.Flatten()
        self.softmax1 = torch.nn.Softmax(dim=-1)
        self.linear1 = torch.nn.Linear(64, 1)
    def forward(self, x1):
        v1 = x1.permute(1, 0, 2)
        v2 = self.lstm1(v1)[0]
        x2 = self.flatten1(v2)
        v3 = self.softmax1(x2)
        x3 = v3.permute(1, 0, 2)
        v4 = torch.nn.functional.linear(x3, self.linear1.weight, self.linear1.bias)
        return v4
# Inputs to the model
x1 = torch.randn(4, 2, 1, 32)
