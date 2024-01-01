
class Model(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embedding = torch.nn.Embedding(100, hidden)

    def forward(self, k, q, v):
        d = self.embedding(k)
        e1 = torch.matmul(q, d.tranpose(-2, -1)) 
        e2 = e1 * 100
        softmax_e2 = torch.softmax(e2, dim=-1)
        dropout_e2 = torch.nn.functional.dropout(softmax_e2, p=0.8)
        output = torch.matmul(dropout_e2, v) 
        return output

# Initializing the model
m = Model(20)

# Inputs to the model
k = torch.randint(0, 100, (1, 15)).long()
q = torch.randn(1, 17, 20)
v = torch.randn(1, 15, 20)
