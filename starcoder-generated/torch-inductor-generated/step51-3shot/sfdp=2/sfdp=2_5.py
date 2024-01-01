
class Model(torch.nn.Module):
    def __init__(self, query, key, value, inv_scale_factor):
        super().__init__()
        self.inv_scale_factor = inv_scale_factor
        self.matmul = torch.nn.Matmul()

    def forward(self, query, key, value, dropout_p):
        qk = self.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
query = torch.randn(1, 5, 8)
key = torch.randn(1, 7, 8)
value = torch.randn(1, 7, 5)
m = Model(query, key, value, 10)
inv_scale_factor = 10
dropout_p = 0.5
