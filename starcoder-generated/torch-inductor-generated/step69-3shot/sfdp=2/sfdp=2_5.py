
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attention_mask):
        query = torch.tensor([[1, 2], [3, 4], [5, 6]])
        key = torch.tensor([[7, 7], [8, 8], [9, 9]])
        value = torch.tensor([[11, 12], [13, 14], [15, 16]])

        scale_factor = torch.sqrt(torch.tensor(query.size(-1)))
        inv_scale_factor = torch.sqrt(torch.tensor(value.size(-1)))

        attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(-2)
        attention_mask = attention_mask.unsqueeze(1)
        
        query = query.unsqueeze(-2)
        query = torch.matmul(attention_mask, query)
        query = torch.reshape(query, (-1, query.size(2), query.size(-1)))

        key = key.unsqueeze(-3)
        key = torch.matmul(attention_mask, key)
        key = torch.reshape(key, (-1, key.size(2), key.size(-1)))

        value = value.unsqueeze(-3)
        value = torch.matmul(attention_mask, value)
        value = torch.reshape(value, (-1, value.size(2), value.size(-1)))

        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)

        out = dropout_qk.matmul(value)
        return out
        
m = Model()

# Inputs to the model
attention_mask = torch.tensor([[[1.0, 0.], [1.0, 1.]]])
