
class Model(torch.nn.Module):
    def __init__(self, q, k, v, m):
        super().__init__()
        self.query = torch.nn.Linear(q, m)
        self.key = torch.nn.Linear(k, m)
        self.value = torch.nn.Linear(v, m)
 
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = (k.shape[-1] ** -0.25)
        scaled_qk = qk / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(v)
        return output, dropout_qk

# Initializing the model
m = Model(256, 512, 512, 1024)

# Inputs to the model
x1 = torch.randn(4, 256)
x2 = torch.randn(4, 512, 12)
__output__, 