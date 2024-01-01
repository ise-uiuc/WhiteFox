
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        self.dropout = torch.nn.Dropout(self.dropout_p)
 
    def forward(self, x1):
        qk = torch.matmul(x1, x2)
        inv_scale_factor = 32768.0
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 32)
x2 = torch.randn(1, 20, 32)
x3 = torch.randn(1, 20, 32)
