
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
 
    def forward(self, input, output):
        q = self.linear(input)
        k = output.transpose(-2, -1)
        qk = torch.matmul(q, k)
        inv_sf = torch.tensor([0.12, 0.08, 0.08, 0.12])
        scaled_qk = qk.mul(inv_sf.view(1, 1, -1, 1))
        softmax_qk = softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.8)
        o = torch.matmul(dropout_qk, output)
        return o

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 4)
output = torch.randn(1, 3, 4)
