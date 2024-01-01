
class Model(torch.nn.Module):
    def __init__(self, query=64, key=16, value=32):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn((query, query)), requires_grad=True)
        self.key = torch.nn.Parameter(torch.randn((key, query)), requires_grad=True)
        self.value = torch.nn.Parameter(torch.randn((value, query)), requires_grad=True)
        self.inv_scale_factor = torch.nn.Parameter(torch.Tensor([200]), requires_grad=True)
        self.dropout = torch.nn.Dropout(p=0.15)
 
    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 32)
