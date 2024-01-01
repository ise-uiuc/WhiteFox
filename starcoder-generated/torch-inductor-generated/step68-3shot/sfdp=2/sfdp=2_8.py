
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 8

        self.query = torch.nn.Parameter(torch.rand(12, 16, self.scale_factor))
        self.key = torch.nn.Parameter(torch.rand(12, 16, self.scale_factor))
        self.value = torch.nn.Parameter(torch.rand(12, 16, 64))

    def forward(self, x1):
        qk = torch.matmul(x1, self.key)
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(self.value)

        return self.out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(12, 16, 64)
