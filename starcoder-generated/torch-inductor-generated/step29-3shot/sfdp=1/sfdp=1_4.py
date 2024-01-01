
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(2, 3, 256, int(256 / 4)))
        self.key = torch.nn.Parameter(torch.randn(2, 3, 256, int(256 / 4)))
        self.value = torch.nn.Parameter(torch.randn(2, 3, 256, int(256 / 4)))
        self.inv_scale_factor = torch.nn.Parameter(torch.rand(2, 1))
        self.dropout_p = torch.nn.Parameter(torch.rand(1))

    def forward(self, x1, mask)
        qk = torch.matmul(x1, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 256, 256)
mask = torch.randn(2, 1, 1, 256)
