
class Model(torch.nn.Module):
    def __init__(self, dropout, scale_factor):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand(4, 512, 128))
        self.key = torch.nn.Parameter(torch.rand(4, 128, 256))
        self.value = torch.nn.Parameter(torch.rand(4, 256, 256))
        self.dropout = dropout
        self.scale_factor = scale_factor

    def forward(self, q, k):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model(0.5, 16)

# Inputs to the model
q = torch.randn(1, 4, 512, 128)
k = torch.randn(1, 4, 256, 256)
