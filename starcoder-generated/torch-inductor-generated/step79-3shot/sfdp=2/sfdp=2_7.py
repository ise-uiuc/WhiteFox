
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        q = x1
        k = x2
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1.0 / math.sqrt(512)
        v = self.value
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 4096)
x2 = torch.randn(1, 512, 4096)
