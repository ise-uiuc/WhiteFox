
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, v1, v2, v3):
        qk = torch.matmul(v1, v2.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v3)
        return output
 
# Model instantiation
scale_factor = torch.nn.Parameter(torch.tensor(1.0))
dropout_p = 0.5
m = Model()
 
# Input tensors
v1 = torch.randn(1, 1, 12, 64)
v2 = torch.randn(1, 1, 64, 512)
v3 = torch.randn(1, 512, 5)
