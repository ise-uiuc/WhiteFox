
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 0.7
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        # dropout
        output = softmax_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 1024)
x2 = torch.randn(1, 16, 512)
