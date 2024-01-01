
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.zeros([4, 5, 10], dtype=torch.float), requires_grad=True)
        self.key = torch.nn.Parameter(torch.zeros([4, 6, 12], dtype=torch.float, requires_grad=True))
        self.value = torch.nn.Parameter(torch.zeros([4, 6, 20], dtype=torch.float, requires_grad=True))
        self.scale_factor = 10

    def forward(self, k1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5, training=True)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
k1 = torch.randn(2, 4, 5, 10)
