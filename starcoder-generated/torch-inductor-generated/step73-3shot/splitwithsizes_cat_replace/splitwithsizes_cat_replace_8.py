
import torch.jit as jit
    
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.features = torch.nn.ModuleList([torch.nn.Linear(16, 32), torch.nn.Hardtanh(inplace=True), torch.nn.Hardtanh(inplace=True)])
        self.features = torch.nn.ModuleList([torch.nn.LSTM(16, 16, 2)])

        ##self.features = torch.nn.ModuleList([torch.nn.LSTM(2, 8, 2), torch.nn.ReLU(), torch.nn.LSTM(8, 8, 2)])
        self.lstm0 = torch.nn.LSTM(3, 8, 2)
        self.lstm1 = torch.nn.LSTM(8, 8, 2)
        self.hardthanh = torch.nn.Hardtanh(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(8, 4)
    def forward(self, input0):
        h0 = torch.rand(1, 2, 2, 8)
        c0 = torch.rand(1, 2, 2, 8)
        seq_lengths = [0, 2]
        model = jit.script(Model())
        lstm_out0, (hn0, cn0) = model.lstm0(input0, (h0, c0))
        # lstm_out0, (hn0, cn0) = self.features[0](input0)

        lstm_out1, (hn1, cn1) = self.lstm1(lstm_out0, (hn0, cn0))
        recurrent_relu_out = self.hardthanh(lstm_out1)

        sigmoid_out = self.sigmoid(recurrent_relu_out)
        flatten_out = self.flatten(sigmoid_out)
        dropout_out = self.dropout(flatten_out)
        linear_out = self.linear(dropout_out)

        return (linear_out, (hn1, cn1))
# Inputs to the model
x1 = torch.randn(10, 3, 16)
