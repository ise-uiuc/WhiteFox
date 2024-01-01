
class Model(torch.nn.Module):
    def forward(self, input_tensor):
        query = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
        key = [[<KEY>], [4, 5, 6, 7], [5, 6, 7, 8]]
        value = [[6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11]]
        attn_weight = input_tensor @ query / math.sqrt(len(query))
        attn_weight += torch.ones_like(attn_weight)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model()

# Inputs for the model
input_tensor = torch.Tensor([[1], [2], [3]])
