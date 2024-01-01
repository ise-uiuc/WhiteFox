
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(16, 10, 3072))
        self.key = torch.nn.Parameter(torch.randn(16, 10, 3072))
        self.value = torch.nn.Parameter(torch.randn(16, 12, 3072))
        self.attn_mask = torch.nn.Parameter(torch.tril(torch.ones(16, 10, 10)).view(16, 10, 10), requires_grad=False)
 
    def forward(self, x2):
        q = self.query[:x2.size(0), :, :].view(x2.size(0), x2.size(1), -1)  # select only the valid query
        k = self.key[:x2.size(0), :, :].transpose(-2, -1)  # select only the valid key
        v = self.value[:x2.size(0), :, :].transpose(-2, -1)  # select only the valid value
        qk = q @ k / math.sqrt(q.size(-1))  # scaled dot-product attention
        qk = qk + self.attn_mask.unsqueeze(1).repeat(1, x2.size(1), 1, 1)  # apply the mask
        attn_weight = torch.softmax(qk, dim=-1)  # apply softmax
        output = attn_weight @ v  # compute the weighted sum
        return output

# Initializing the model
m2 = Model()

# Input to the model
x2 = torch.randn(16, 10, 3072)
output = m2(x2)  # you can compute the output for any input batch
torch.save(m2.state_dict(), "m2.pt")
torch.onnx.export(m2, x2, "m2.onnx", opset_version=11)

