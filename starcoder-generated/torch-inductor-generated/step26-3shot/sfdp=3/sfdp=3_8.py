
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
 
    def forward(self, q, k, v, scale_factor=1.0):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the new model
m = Model()

# Inputs to the model
q = torch.randn(1, 16, 256)
k = torch.randn(1, 16, 256)
v = torch.randn(1, 16, 256)
