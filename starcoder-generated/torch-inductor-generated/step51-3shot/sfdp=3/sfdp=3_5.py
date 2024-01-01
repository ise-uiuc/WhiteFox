
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, k1, q1, v1):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
k1 = torch.randn(1, 120, 32)
q1 = torch.randn(1, 150, 32)
v1 = torch.randn(1, 130, 24)
