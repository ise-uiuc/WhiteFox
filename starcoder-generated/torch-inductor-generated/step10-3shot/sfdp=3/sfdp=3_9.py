
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qk_scale_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.dropout_p = 0.001
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.qk_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(128, 12, 1024)
k = torch.randn(128, 12, 1024)
v = torch.randn(128, 12, 1024)

