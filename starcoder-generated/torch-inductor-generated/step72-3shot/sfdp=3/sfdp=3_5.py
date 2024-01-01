
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, v1, v2, v3):
        qk = torch.matmul(v1, v2.transpose(-2, -1))
        scaled_qk = qk.mul(0.7)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.8)
        output = dropout_qk.matmul(v3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 8, 64, 64)
v2 = torch.randn(1, 8, 64, 64)
v3 = torch.randn(1, 8, 64, 64)
