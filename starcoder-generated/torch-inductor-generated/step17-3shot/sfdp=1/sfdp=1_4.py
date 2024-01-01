 with two different inputs where one is used to initialize the parameters of the second
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(65536, 256)

    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(0, 1))
        scale_factor = torch.sqrt(torch.tensor(842137.5))
        inv_scale_factor = torch.tensor(1) / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = softmax_qk.matmul(v)
        v1 = self.lin(output)
        return v1

# Inputs to the model
q = torch.randn(842137, 256)
k = torch.randn(842137, 256)
v = torch.randn(842137, 256)
