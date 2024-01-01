
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = query.shape[-1] ** 0.5
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = F.dropout(softmax_qk, training=self.training)
        output = torch.matmul(dropout_qk, value)

        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(100, 50, 768)
key = torch.randn(100, 12, 768)
value = torch.randn(100, 12, 768)
