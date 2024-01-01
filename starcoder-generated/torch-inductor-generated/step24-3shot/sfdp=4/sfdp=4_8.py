
class Model(torch.nn.Module):
    def forward(self, x1):
        attention_mask = generate_attention_mask(x1, x1)
        k1 = torch.exp(torch.randn(x1.shape[1], x1.shape[1]))
        q1 = torch.exp(torch.randn(x1.shape[1], x1.shape[1]))
        v1 = torch.exp(torch.randn(x1.shape[1], x1.shape[1]))
        out = torch.matmul(q1, k1) / math.sqrt(q1.shape[1])
        out *= attention_mask
        attn_weight = torch.softmax(out, dim=1)
        x2 = torch.bmm(attn_weight, v1)
        return x2

def generate_attention_mask(input, output):
    return torch.tril(torch.ones(input.shape[1], output.shape[1]))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
