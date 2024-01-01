
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(6, 10)
        self.k = torch.nn.Linear(6, 10)
        self.v = torch.nn.Linear(6, 10)
 
    def forward(self, query, key, value, dropout_p, scale_factor):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        logits = torch.matmul(q, k.transpose(-2, -1) / scale_factor)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output = torch.nn.functional.dropout(probs, p=dropout_p)
        output = torch.matmul(output, v)
        return output


# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 5, 6)
key = torch.randn(1, 5, 6)
value = torch.randn(1, 5, 6)
dropout_p = 0.1
scale_factor = torch.tensor([1e-4])
