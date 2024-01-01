
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.0)
        self.matmul = torch.matmul
 
    def forward(self, q1, k1, v1):
        qk = self.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.div(10.0)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 8, 64, 64)
k1 = torch.randn(1, 8, 64, 64)
v1 = torch.randn(1, 8, 64, 64)
