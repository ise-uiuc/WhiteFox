

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(8, 8, 10))
        self.key = torch.nn.Parameter(torch.randn(8, 8, 15))
        self.value = torch.nn.Parameter(torch.randn(8, 8, 20))

    def forward(self, x, dropout_p=0.1):
        q = self.query
        k = self.key
        v = self.value
        scale_factor = torch.tensor(15)
        inv_scale_factor = torch.div(scale_factor, torch.tensor(1e-8))
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 8, 10)
