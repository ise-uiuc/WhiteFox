
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.16805113117317414)
        self.matmul = torch.nn.Linear(300, 128)
 
    def forward(self, input, weight):
        qk = torch.matmul(input, weight.transpose(-2, -1))
        scaled_qk = qk.div(1)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(1, 300)
x2 = torch.randn(300, 128)
