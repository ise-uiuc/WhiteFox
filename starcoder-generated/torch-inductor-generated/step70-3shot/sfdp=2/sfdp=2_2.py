
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 8
        self.query = torch.nn.Parameter(torch.randn(hidden_size, 8) * 0.01)
        self.key = torch.nn.Parameter(torch.randn(hidden_size, 8) * 0.01)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, self.key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 8)
x2 = torch.randn(1, 8, 1)
