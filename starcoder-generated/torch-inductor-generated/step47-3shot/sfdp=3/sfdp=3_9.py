
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(42,84))
        self.key = torch.nn.Parameter(torch.randn(17,84))
        self.value = torch.nn.Parameter(torch.randn(17,42))
 
    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scale_factor = self.key.size(-1) ** -0.2
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 42)
