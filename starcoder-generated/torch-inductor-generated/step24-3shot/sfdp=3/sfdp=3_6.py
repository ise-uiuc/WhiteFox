
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
     
    def forward(self, m1, m2):
        qk = torch.matmul(m1, m2.transpose(-2, -1))
        scale_factor = torch.tensor.type(torch.float32).pow(
            2.0 / (m1.shape[-1] ** 0.5))
        dropout_p = 0.2
        output = torch.nn.functional.dropout(
            qk.mul(scale_factor).softmax(dim=-1), p=dropout_p).matmul(m2)
        return output

# Initializing the model
m1 = torch.nn.init.uniform_(torch.Tensor(3, 4))
m2 = torch.nn.init.uniform_(torch.Tensor(4, 3))
