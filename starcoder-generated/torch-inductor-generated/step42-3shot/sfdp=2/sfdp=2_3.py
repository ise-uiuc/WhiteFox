
class Model(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, queries, keys, values, dropout_p=0.2, inv_scale_factor=1.0):
        qk = torch.matmul(queries, keys.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(values)
        return output

# Initializing the model
m = Model(1000)

# Inputs to the model
queries = torch.randn(64, 10, 100)
keys = torch.randn(64, 20, 100)
values = torch.randn(64, 20, 100)
