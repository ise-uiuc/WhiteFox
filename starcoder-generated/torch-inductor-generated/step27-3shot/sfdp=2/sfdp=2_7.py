
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale_factor = 1.0 / math.sqrt(832)

    def forward(self, queries, keys, values, mask, dropout):
        qk = torch.matmul(queries, keys.transpose(-2, -1)) * self.inv_scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout)
        output = dropout_qk.matmul(values).mean(dim=1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(1, 2048, 768)
keys = torch.randn(1, 10, 2048, 768)
values = torch.randn(1, 10, 2048, 768)
mask = torch.ones(1, 10, 1, 10) > 0
dropout = 0.0
