
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        scale=1/sqrt(2),
        inv_scale_factor=sqrt(2),
        dropout_p=0.5
    ):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initialzing the model
m = Model()

# Inputs of the model
q = torch.randn(1, 8, 5, 5)
k = torch.randn(1, 8, 10, 10)
v = torch.randn(1, 8, 10, 10)
