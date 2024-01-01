
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = float(q.shape[-1] / k.shape[-2])
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output
 
# Initializing the model
m = Model(512, 256)

# Inputs to the model
x2 = torch.randn(1, 256, 8)
x3 = torch.randn(1, 256, 8)
x4 = torch.randn(1, 512, 8)
