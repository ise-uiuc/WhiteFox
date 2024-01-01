
class Model(torch.nn.Module):
    def get_inv_scale(self):
        return math.sqrt(self.__key_dim__)
 
    def __init__(self):
        super().__init__()
        self.__key_dim__ = 128
        self.projection_q = torch.nn.Linear(self.__key_dim__, self.__key_dim__)
        self.projection_k = torch.nn.Linear(self.__key_dim__, self.__key_dim__)
        self.projection_v = torch.nn.Linear(self.__key_dim__, self.__key_dim__)
        self.projection_o = torch.nn.Linear(self.__key_dim__, self.__key_dim__)
 
    def forward(self, x1, x2):
        v1 = self.projection_q(x1)
        v2 = self.projection_k(x2)
        v3 = self.projection_v(x2)
        v4 = torch.matmul(v1, v2.transpose(-2, -1)) / self.get_inv_scale()
        v5 = v4.softmax(-1)
        v6 = torch.matmul(v5, v3)
        v7 = self.projection_o(v6)
        return v7

# Initialize the model
m = Model()

# The model inputs are the tensors representing the queries and keys
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 128, 128)
