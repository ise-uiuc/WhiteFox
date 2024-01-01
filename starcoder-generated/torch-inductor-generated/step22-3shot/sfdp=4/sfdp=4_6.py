
class Attention_mask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, attn_mask):
        ctx.save_for_backward(query, key, value)
        ctx.attn_mask = attn_mask
        attn_weight = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(query.shape[-1])
        #attn_weight = attn_weight + attn_mask
        #attn_weight.masked_fill_(attn_mask, -float('inf'))
        return attn_weight.masked_fill(attn_mask, -float("inf"))
    
    @staticmethod
    def backward(ctx, grad):
        query, key, value = ctx.saved_tensors
        attn_mask = ctx.attn_mask
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        attn_weight = attn_weight + attn_mask
        attn_weight.masked_fill_(attn_mask, -float('inf'))
        attn_weight = torch.softmax(attn_weight, dim=-2)
        attn_weight = attn_weight.masked_fill(attn_mask, 0.0)
        attn_grad1 = torch.matmul(attn_weight, grad)
        attn_grad2 = torch.matmul(grad, attn_weight.transpose(0, 1))
        attn_grad2 = attn_grad2.transpose(0, 1)
        query_grad = torch.matmul(attn_grad1, value)
        key_grad = torch.matmul(attn_grad2, value.transpose(0, 1))
        query_grad = query_grad / query.shape[-1]
        key_grad = key_grad / query.shape[-1]
        
        #attn_grad2 = attn_weight.transpose(0, 1)
        #attn_grad2 = torch.matmul(grad, attn_grad2)
        #attn_grad2 = attn_grad2.transpose(0, 1)
        return query_grad, key_grad, attn_grad2, None

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_mask = torch.zeros(64, 64).bool().cuda()

    def forward(self, query, key, value):
        attn_weight1 = Attention_mask.apply(query, key, value, self.attn_mask).cuda()
        attn_weight2 = attn_weight1.masked_fill(self.attn_mask, 0.0)
        return attn_weight2

# Initializing the model
model = Model()

# Inputs to the model
query = torch.randn(64, 3, 16, 16)
key = torch.randn(64, 3, 16, 16)
value = torch.randn(64, 4, 16, 16)
