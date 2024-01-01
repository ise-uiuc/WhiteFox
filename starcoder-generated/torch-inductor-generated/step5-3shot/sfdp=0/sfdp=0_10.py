
class Attention(torch.nn.Module):
    def __init__(self, d_q, d_k, d_v):
        super(Attention, self).__init__()
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, query, key, value):
        d_k, d_v = self.d_k, self.d_v
        # Perform matrix multiplactions
        query = query.view(-1, query.shape[-2], d_q)
        key = key.view(-1, key.shape[-2], d_k)
        value = value.view(-1, value.shape[-2], d_v)
        matmul = torch.matmul(query, key.transpose(-2, -1))
        inv_sqrt = 1 / math.sqrt(self.d_k) # get the square root of d_k
        matmul = matmul * inv_sqrt # divide each element by the square root of d_k
        attention_parameters = matmul.softmax(dim=-1) # get the attention parameters
        # Output
        output = attention_parameters.matmul(value)
        return output

# Initializing the model
attention = Attention(d_q=d_q, d_k=d_k, d_v=d_v)

# Inputs to the model
