
class Model(torch.nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.scale_factor = (self.n_heads * self.n_heads) **.5
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = torch.matmul(dropout_qk, x2)
        return output


# Initializing the model
m = Model(8) # n_heads = 8

# Inputs to the model
x1 = torch.randn(1, 8, 9, 10)
x2 = torch.randn(1, 9, 4, 5) # x1 and x2 are in different shapes. Please transpose x1 to get a shape like x2.
