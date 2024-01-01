
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, q, k, v, scale_factor):
        qk = q.matmul(k.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = self.softmax.forward(scaled_qk)
        dropout_qk = self.dropout.forward(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
model = Model()

# Inputs to the model
q = torch.randn(1, 16, 32)
k = torch.randn(1, 16, 128)
v = torch.randn(1, 16, 128)
