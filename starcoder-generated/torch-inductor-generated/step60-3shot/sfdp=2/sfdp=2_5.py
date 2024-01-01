
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.6)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(0.125)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(4, 2, 8)
k = torch.randn(4, 2, 8)
v = torch.randn(4, 2, 8)
