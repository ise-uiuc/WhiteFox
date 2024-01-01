
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        self.scale_factor = np.sqrt(32 / (0.0212 * 8))

    def forward(self, x1, x2):
        q1 = torch.nn.functional.linear(x1, x2)
        k1 = torch.nn.functional.linear(x1, x2)
        v1 = torch.nn.functional.tanh(torch.nn.functional.linear(x1, x2))
        qk1 = torch.matmul(q1, k1.transpose(-2, -1))
        scaled_qk1 = qk1.mul(self.scale_factor)
        dropout_qk1 = torch.nn.functional.dropout(
            scaled_qk1.softmax(dim=-1), p=self.dropout_p)
        output1 = dropout_qk1.matmul(v1)
        return output1

# Initializing the model
m = Model()
n = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
x2 = torch.randn(32, 32)
__output1__ = m(x1, x2)
__output2__ = n(x1, x2)

