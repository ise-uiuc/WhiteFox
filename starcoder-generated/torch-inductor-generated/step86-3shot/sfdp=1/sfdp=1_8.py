
class Model(torch.nn.Module):
    def __init__(self, num_heads, dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p

    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax = scaled_qk.softmax(dim=-1)
        dropout_softmax = torch.nn.functional.dropout(softmax, p=self.dropout_p)
        output = dropout_softmax.matmul(value)
        return output

# Initializing the model
m = Model(num_heads, dropout_p)

# Inputs to the model
query = torch.randn(1, 8, 512, 512)
key = torch.randn(1, 8, 512, 512)
value = torch.randn(1, 8, 512, 512)
inv_scale_factor = torch.randn(1, 1, 1)
