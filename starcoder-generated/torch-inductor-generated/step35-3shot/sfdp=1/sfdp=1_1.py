
__parameters__ = [["query", "key", "value"], [1, 2, 3], [4, 5, 6], [1, 10, 10]]
__input_tensor__ = torch.randn(1, 10, 10)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(value)
        return output

# Testing the model
m = Model()
q = torch.randn(1, 4, 10)
k = torch.randn(1, 5, 10)
v = torch.randn(1, 6, 10)
inv_scale_factor = 1e4
