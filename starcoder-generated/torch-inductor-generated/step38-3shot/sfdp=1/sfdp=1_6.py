
class Model(torch.nn.Module):
    def _init_(self, model_dimension, dropout_probability, scale_factor):
        super.__init__()
        self.query = torch.nn.Linear(model_dimension, model_dimension)
        self.key = torch.nn.Linear(model_dimension, model_dimension)
        self.value = torch.nn.Linear(model_dimension, model_dimension)
        self.dropout_p = dropout_probability
        self.inv_scale_factor = 1. / scale_factor
 
    def forward(self, query_tensor, key_tensor, value_tensor):
        query = self.query(query_tensor)
        key = self.key(key_tensor)
        value = self.value(value_tensor)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(model_dimension=256, dropout_probability=0.1, scale_factor=100.)

# Inputs to the model
query_tensor = torch.randn(32, 256, 512)
key_tensor = torch.randn(32, 256, 512)
value_tensor = torch.randn(32, 256, 512)
