
from torch.onnx import _add_lstm_onnx_node, _set_rnn_dropout, _set_rnn_dropout_state
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(1, 2, bias=True, dropout=0.3)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.rnn(v1)
        return v2[0].detach()
# Inputs to the model
x1 = torch.randn(1, 2, 2)
