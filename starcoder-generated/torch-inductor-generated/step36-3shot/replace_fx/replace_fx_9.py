
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.nn.utils.rnn.SequentialRNNCell(torch.nn.utils.rnn.LSTMCell(2, 2))
    def forward(self, x1):
        x2 = x1.transpose(0, 1)  
        x3, x4 = self.x1(x2)
        x5 = x1 + x3
        return x5
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.nn.utils.rnn.RNN(cell_class=torch.nn.LSTMCell, input_size=2, hidden_size=2, num_layers=1, batch_first=True, dropout=0, bidirectional=False)
    def forward(self, x1):
        x1, x2 = x1.shape
        x1 = x1.unsqueeze(0)
        x3 = self.x1(x1, torch.zeros(x2, 2))
        x4 = x3[0][0]
        return x4
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.nn.utils.rnn.RNN(cell_class=torch.nn.LSTMCell, input_size=2, hidden_size=2, num_layers=1, batch_first=True, dropout=0, bidirectional=False)
    def forward(self, x1):
        x2 = x1.shape
        x2 = x2[1]
        x3 = x1.unsqueeze(0)
        x3 = self.x1(x3, torch.zeros(1, x2, 2))
        x4 = x3[0][0]
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
