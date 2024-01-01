
class Model(torch.nn.Module):
    def __init__(self):
        self.query = torch.nn.Parameter(torch.rand(2, 4, 16))
        self.key = torch.nn.Parameter(torch.rand(2, 16, 64))
        self.value = torch.nn.Parameter(torch.rand(2, 16, 32))
        self.scale_factor = 2.463637842767293
        self.dropout_p = 0.41562485721199752
 
    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk / self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4, 16)
