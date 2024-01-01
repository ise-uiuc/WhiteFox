
class Model(torch.nn.Module):
    def __init__(self, d_model, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
d_model = 1024
dropout_p = 0.2
dropoutTied = Model(d_model, dropout_p)

# Inputs to the model
query = torch.randn(size=(1, 128, d_model))
key = torch.randn(size=(1, 32, d_model))
value = torch.randn(size=(1, 32, d_model))
inv_scale_factor = 1.0 / math.sqrt(d_model)
