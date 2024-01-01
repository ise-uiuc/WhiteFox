
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.p = dropout_p
 
    def forward(self, x1, x2, x3, x4):
        w1 = torch.matmul(x1, x2.transpose(-2, -1))
        w2 = w1 / 2
        w3 = w1.softmax(dim=-1)
        w4 = nn.functional.dropout(w3, p=self.p)
        w5 = torch.matmul(w4, x3)
        w6 = torch.matmul(w2, w5)
        w7 = torch.matmul(w6, x4)
        return w7

# Initializing the model
m = Model()
# Inputs to the model
input_scale_factor = 2
key = torch.randn(1, 197, 768)
query = torch.randn(1, 256, 768)
value = torch.randn(1, 256, 768)
input_inv_scale_factor = 1 / input_scale_factor
