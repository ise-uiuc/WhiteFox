
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__init_attr_0__ = torch.tensor(1, dtype=torch.float32)
        self.__init_attr_1__ = torch.tensor(1.0, dtype=torch.float32)
 
    def forward(self, query, key, value):
        op_0 = torch.matmul(query, key.transpose(-2, -1))
        op_1 = torch.div(op_0, self.__init_attr_0__)
        op_2 = torch.softmax(op_1, dim=-1)
        op_3 = torch.nn.functional.dropout(op_2, p=self.__init_attr_1__)
        op_4 = torch.matmul(op_3, value)
        return op_4

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 2, 2)
key = torch.randn(2, 3, 2, 2)
value = torch.randn(2, 3, 2, 2)
