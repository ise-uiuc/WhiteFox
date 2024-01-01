
class Model(torch.nn.Module):
    def __init__(self, p=0.4, inv_scale_factor=256):
        super().__init__()
        self.dropout_p = p
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 512, 1024)
k = torch.randn(1, 512, 1024)
v = torch.randn(1, 512, 1024)
