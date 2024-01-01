
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(10.0)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 5, 10)
x2 = torch.randn(4, 7, 10)
