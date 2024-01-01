
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.tensor(10000.0)
 
    def forward(self, q1, k1, v1):
        qk = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.0)
        output = dropout_qk.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(2, 16, 4096)
k1 = torch.randn(2, 16, 2048)
v1 = torch.randn(2, 16, 2048)
