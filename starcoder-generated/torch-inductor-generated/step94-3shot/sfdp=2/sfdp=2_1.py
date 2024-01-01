
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.softmax = torch.nn.Softmax(dim=1)
 
    def forward(self, q, k):
        # v1 = torch.matmul(k, q.transpose(0, 1))
        # v2 = v1.div(16)
        # v3 = self.softmax(v2)
        # v4 = torch.nn.functional.dropout(v3, p=0.2)
        # output = v4.matmul(k)
        output = torch.matmul(torch.matmul(k, q.transpose(0, 1)), q)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(128, 512)
k = torch.randn(128, 512)
