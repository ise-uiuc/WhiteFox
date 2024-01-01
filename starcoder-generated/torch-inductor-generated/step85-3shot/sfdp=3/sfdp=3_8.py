
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.Tensor([3.0]))
        self.dropout_p = torch.nn.Parameter(torch.linspace(0.0, 1.0, 100))
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Input to the model
q = torch.randn(1, 8, 16, 16)
k = torch.randn(1, 16, 32, 32)
v = torch.randn(1, 16, 32, 32)
