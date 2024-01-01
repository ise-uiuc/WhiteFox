
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(0.5)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scale_factor = 1 / math.sqrt(4)
        inv_scale_factor = 1 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(x1)

        return scale_factor

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 4)
