
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k1, v1, scale_factor, dropout_p):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
model = Model()

# Inputs to the model
q1 = torch.randn(1, 3, 64, 64)
k1 = torch.randn(1, 3, 64, 64)
v1 = torch.randn(1, 16, 64, 64)
scale_factor = torch.tensor(1)
dropout_p = torch.tensor(0.5)
