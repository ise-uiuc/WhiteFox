
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1

    def forward(self, qk, inv_scale_factor):
        softmax_qk = self.dropout_qk(qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
qk = torch.randn(5, 10, 64, 64)
inv_scale_factor = torch.randn(1)
