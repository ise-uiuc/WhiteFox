
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_qk = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, q, k, v, scale_factor):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout_qk(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 12, 64, 64)
k = torch.randn(1, 12, 64, 64)
v = torch.randn(1, 12, 64, 64)
scale_factor = torch.randn(1)
