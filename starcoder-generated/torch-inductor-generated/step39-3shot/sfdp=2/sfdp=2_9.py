
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = torch.nn.Linear(16, 32)
        self.linear_k = torch.nn.Linear(20, 32)
        self.linear_v = torch.nn.Linear(4, 32)
 
    def forward(self, x1, x2):
        q = self.linear_q(x1).unsqueeze(1)
        v = self.linear_v(x2)
        k = self.linear_k(x2).transpose(-2, -1)
        scaled_q = q
        softmax_q = scaled_q.softmax(dim=-1)
        dropout_q = torch.nn.functional.dropout(softmax_q, p=0.5)
        output = dropout_q.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 16)
x2 = torch.randn(20, 4)
