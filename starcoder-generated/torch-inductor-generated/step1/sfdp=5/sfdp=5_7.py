 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(5, 5)
        self.dropout = torch.nn.Dropout(1)

    def forward(self, query, key, value, masks):
        qkv_matrix = self.qkv(query)
        qk_matrix = qkv_matrix.split([3, 4, 5], dim=1)

        scaling_factor = 1 / torch.sqrt(torch.to_tensor(5))

        attention_logits = (qk_matrix[0] @ qk_matrix[1].transpose(1, 2)) * scaling_factor + masks
        attention_weights = torch.softmax(attention_logits, dim=-1).unsqueeze(1)
        attentioned_value = attention_weights * qk_matrix[2]
        dropped = self.dropout(attentioned_value)
        output_embedding = torch.bmm(dropped, value)
        return output_embedding

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 3, 5)
key = torch.randn(2, 4, 5)
value = torch.randn(2, 4, 6)
masks = torch.tensor([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                      [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]])
