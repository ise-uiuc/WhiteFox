
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.parameter.Parameter(torch.randn(4, 5, 6, 8))
        self.k = torch.nn.parameter.Parameter(torch.randn(4, 6, 8, 10))
        self.v = torch.nn.parameter.Parameter(torch.randn(4, 6, 8, 10))

    def forward(self, query):
        q = self.q
        k = self.k
        v = self.v
        inv_scale_factor = 0.1
        dropout_p = 0.5
        qk = torch.matmul(query, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 5, 6, 8)
