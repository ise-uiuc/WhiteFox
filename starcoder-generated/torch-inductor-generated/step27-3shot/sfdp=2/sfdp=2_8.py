
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.query = torch.randn(4, 13, 64)
        self.key = torch.randn(4, 17, 64)
        self.value = torch.randn(4, 17, 64)

    def forward(self, query, key, value, dropout_p):
        inv_scale_factor = 0.1
        self.softmax_qk = torch.matmul(
            query,
            key.transpose(-2, -1)).div(
            inv_scale_factor).softmax(dim=-1)
        self.dropout_qk = torch.nn.functional.dropout(
            self.softmax_qk, p=dropout_p)
        output = torch.matmul(self.dropout_qk, value)
        return output

# Initializing the model
m = Model(num_heads=4)

# Inputs to the model
query = torch.randn(4, 7, 64)
key = torch.randn(4, 5, 64)
value = torch.randn(4, 5, 64)
dropout_p = 0.1
