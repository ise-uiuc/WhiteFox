
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        self.softmax_args = {'dim': -1}
        self.dropout_args = {'p': self.dropout_p}
        self.hidden_size = 8
  
    def forward(self, q, k, v):
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)).div(self.hidden_size ** -0.5)
        softmax_qk = scaled_qk.softmax(**(self.softmax_args))
        dropout_qk = torch.nn.functional.dropout(softmax_qk, **(self.dropout_args))
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model for inference
q = torch.randn(1, 8, 64, 24)
k = torch.randn(1, 8, 24, 16)
v = torch.randn(1, 8, 24, 16)
