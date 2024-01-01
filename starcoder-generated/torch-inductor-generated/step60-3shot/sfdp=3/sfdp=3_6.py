
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.parameter.Parameter(torch.randn((8, 24, 14)))
        self.query = torch.nn.parameter.Parameter(torch.randn((8, 32, 18)))
        self.value = torch.nn.parameter.Parameter(torch.randn((8, 24, 18)))
        self.scale_factor =...

    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 18)
