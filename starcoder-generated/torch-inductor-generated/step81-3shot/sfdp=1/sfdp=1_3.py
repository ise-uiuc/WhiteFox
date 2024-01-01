
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1

    def forward(self, query, key, value, scale_factor, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 512, 512)
key = torch.randn(1, 512, 512)
value = torch.randn(1, 512, 512)
scale_factor = 0.5
inv_scale_factor = 1/scale_factor
m6 = m(query, key, value, scale_factor, inv_scale_factor)

