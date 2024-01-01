
class Model(torch.nn.Module):
    def __init__(self, query=3, key=4, value=3):
        super().__init__()
        self.scale_factor = query * key
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout()
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(4, 3)
x3 = torch.randn(3, 4)
