
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
 
    def forward(self, input, w):
        k = torch.einsum("ijk,lkj->ijl", input, w)
        k = k / (self.dropout_p + inp[0].sum(dim=-1, keepdim=True))
        v = torch.einsum("ijk,lkj->ijl", input, w)
        q = torch.einsum("ijk,ijl->ijl", input, w)
        q = q / (self.dropout_p + inp[1].sum(dim=-1, keepdim=True))
        mask = q.new(q.shape[0], q.shape[1], q.shape[1]).fill_(float('-inf')).triu(diagonal=1) 
        mask = mask.cuda()
        return torch.softmax(q + mask, dim=-1).matmul(v)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.rand(3, 50, 10)
k = torch.rand(3, 7, 10)
v = torch.rand(3, 7, 10)
