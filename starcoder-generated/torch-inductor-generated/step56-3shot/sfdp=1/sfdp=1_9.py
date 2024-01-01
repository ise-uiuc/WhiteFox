
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.9
        self.scale_factor = 8

    def forward(self, query, key, value):
        inv_scale_factor = 1 / self.scale_factor
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Input to the model
query = torch.randn(1, 12, 768)
key = torch.randn(1, 20, 768)
value = torch.randn(1, 20, 768)
