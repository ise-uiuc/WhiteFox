
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, dropout_p=0.):
        k = torch.transpose(k, -2, -1)
        qk = torch.matmul(q, k)
        scaled_qk = qk / self.inver_scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk, -1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 64)
k = torch.randn(1, 16, 64)
v = torch.randn(1, 16, 512)
