
class Model(torch.nn.Module):
    def __init__(self, num_heads=1, num_inputs_per_head=1):
        super().__init__()
        self.query = torch.nn.Linear(num_inputs_per_head, num_inputs_per_head)
        self.key = torch.nn.Linear(num_inputs_per_head, num_inputs_per_head)
        self.value = torch.nn.Parameter(torch.randn(num_inputs_per_head, num_inputs_per_head, num_heads))
        self.query.weight = self.key.weight.t()
 
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value
        s = torch.matmul(q, k.t())
        inv_scale_factor = math.sqrt(math.sqrt(q.shape[-1]))
        s = s.div(inv_scale_factor)
        w = s.softmax(dim=-1)
        d = torch.nn.functional.dropout(w, p=0.1)
        o = torch.matmul(d, v).permute(2, 0, 1)
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 768) 
x2 = torch.randn(1, 768)
