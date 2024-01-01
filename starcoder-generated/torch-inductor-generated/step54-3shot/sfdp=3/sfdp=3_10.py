
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_qk_scale_factor = 5

    def forward(x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk * self.softmax_qk_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 4)
x2 = torch.randn(1, 3, 4)
