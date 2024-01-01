
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale = 1 / 0.070576
        scaled_qk = qk * inv_scale
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 9, 85)
x2 = torch.randn(4, 85, 71)
x3 = torch.randn(4, 71, 117)
x4 = torch.randn(4, 117, 110)
