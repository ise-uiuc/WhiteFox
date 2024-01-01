
class Model(torch.nn.Module):
    def forward(self, query, key, value):
        inv_scale_factor = torch.tensor([64, 128, 32, 64, 32, 16, 8, 4, 32, 16, 8, 4]) # (num_heads)
        dropout_p = torch.nn.functional.hardtanh(torch.tensor([0.5])) # The dropout probability between 0 and 1
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(6, 2, 512)
key = torch.randn(6, 3, 512)
value = torch.randn(6, 3, 512)
