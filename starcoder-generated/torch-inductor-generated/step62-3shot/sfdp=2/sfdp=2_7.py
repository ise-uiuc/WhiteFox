
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(3072, 6048)
 
    def forward(self, x1):
        qk = torch.matmul(x1, self.weight.transpose(-2, -1))
        scaled_qk = qk * inv_scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, self.weight)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 4096)
