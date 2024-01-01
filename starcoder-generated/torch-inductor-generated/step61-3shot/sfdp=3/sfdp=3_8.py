
class Model(torch.nn.Module):
    def __init__(self, d_k=16, d_v=16):
        super().__init__()
        self.scale_factor = (d_k ** -0.5)
 
    def forward(self, q1, k1, v1, dropout_p=0.1):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 3, 1024)
k1 = torch.randn(1, 3, 1024)
v1 = torch.randn(1, 3, 1024)
