
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2)
        scale_factor = torch.exp(torch.randn(1))
        softmax_qk = qk.mul(scale_factor)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.7)
        output = dropout_qk.matmul(x1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
x2 = torch.randn(5, 3, 10)
__ouput__ = m(x1, x2)

