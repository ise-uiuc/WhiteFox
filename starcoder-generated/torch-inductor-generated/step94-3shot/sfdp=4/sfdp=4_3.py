
class Model(torch.nn.Module): # use the same format to generate the model
    def __init__(self, query, keys, values, mask):
#         query = torch.randn(1, 64, 56, 56) 
#         keys = torch.randn(1, 64, 56, 56)
#         values = torch.randn(1, 64, 56, 56)
#         mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)       
        super().__init__()
        self.M = torch.nn.Softmax(dim = -1) # use the same format to declare the layer to use the operation
    def forward(self, query, keys, values, mask):
        qk = query @ keys.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + mask
        attn_weight = self.M(qk) # use the same format to add the layer that you use
        output = attn_weight @ values
        return output
# Inputs to the model
query = torch.randn(1, 64, 56, 56)
keys = torch.randn(1, 64, 56, 56)
values = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
