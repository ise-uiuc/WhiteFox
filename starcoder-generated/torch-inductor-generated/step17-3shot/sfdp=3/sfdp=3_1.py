
class Model(torch.nn.Module):
    def __init__(self):
    super().__init__()
    self.qfc = torch.nn.Linear(768, 768)
    self.kfc = torch.nn.Linear(768, 768)
    self.vfc = torch.nn.Linear(768, 768)
    self.ofc = torch.nn.Linear(768, 768)
 
    def forward(self, args):
        q = query.size(0)
        k = key.size(0)
        v = value.size(0)
        query = query.view(q, 768, 1)
        key = key.view(k, 768, 1)
        value = value.view(v, 768, 1)
        q = self.qfc(query)
        k = self.kfc(key)
        v = self.vfc(value)
        scale_factor = (key.view(k, q) * self.scale).softmax(dim=-1)
        softmax_qk = (q.view(q, k) * k.view(k, q) * scale_factor).softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = (dropout_qk.view(q, k) * v.view(v, k)).view(q, 768)
        output = self.ofc(output)
        return (x1)
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 768)
key = torch.randn(1, 3, 768)
value = torch.randn(1, 3, 768)
