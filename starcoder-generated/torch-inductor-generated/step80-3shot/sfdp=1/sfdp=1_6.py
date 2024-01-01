
class Model(torch.nn.Module):
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 256, 48)
key = torch.randn(1, 256, 24)
value = torch.randn(1, 256, 24)
inv_scale_factor = 1.0 / np.sqrt(48)
dropout_p = 0.062777045344360835
