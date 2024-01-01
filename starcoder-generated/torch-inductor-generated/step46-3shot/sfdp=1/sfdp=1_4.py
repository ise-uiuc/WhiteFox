
class Model(torch.nn.Module):
    def __init__(self, query_value_dimension):
        super().__init__()
        self.qkv = torch.nn.Linear(query_value_dimension, 3 * query_value_dimension, bias=False)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        s = self.qkv(query).shape
        scale_factor = torch.sqrt(torch.tensor(s[-1], dtype=torch.float))
        inv_scale_factor = 1. / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(value)
        return output

query = torch.tensor([[-0.7358642204284668, -0.9455091905593872]])
key = torch.tensor([[-0.8529059023857117, 0.3162973244142532]])
value = torch.tensor([[0.400022784280777, 0.45862865925788884]])
