
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        q = self.conv1(x1)
        k = self.conv2(x1)
        v = self.conv3(x1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 16.
        scale_factor = 1 / pow(2., inv_scale_factor)
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        p = 0
        dropout_qk = torch.nn.functional.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
