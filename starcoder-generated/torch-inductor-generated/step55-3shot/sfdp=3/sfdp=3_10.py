
class Model(torch.nn.Module):
    def __init__(self, scale_factor=1, dropout_p=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(scale_factor=1, dropout_p=0.1)

# Inputs to the model
q = torch.randn(1, 4, 64)
k = torch.randn(1, 6, 64)
v = torch.randn(1, 6, 128)
