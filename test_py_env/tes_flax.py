import jax
import jax.numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# 初始化模型参数
model = MLP()
key = jax.random.PRNGKey(0)
x = jnp.ones((1, 32))  # 输入维度为32
params = model.init(key, x)

# 前向推理
output = model.apply(params, x)
print(output)
