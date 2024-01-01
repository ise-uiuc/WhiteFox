
class Model(torch.nn.Module):
    def forward(self, query, key, value, dropout_p, inv_scale_factor):
        scaled_qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = scaled_qk / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 25, 20)
key = torch.randn(1, 20, 40)
value = torch.randn(1, 20, 40)
dropout_p = 0.2
inv_scale_factor = 1 / math.sqrt(20)
