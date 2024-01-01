
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, k1, q2):
        qk = torch.matmul(q2, k1.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        dropout_qk = self.dropout(self.softmax(scaled_qk))
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
k1 = torch.randn(1, 512, 1000)
q2 = torch.randn(1, 512, 30)
