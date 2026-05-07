from pydantic import BaseModel

from model import model as m

class Layer(BaseModel):
    def __init__(self,
        W=None,b=None,
        stride=None, padding=None,
        pooling_size=None, pooling_stride=None
    ):
        self.W = W
        self.b = b
        self.stride = stride
        self.padding = padding
        self.pool_size = pooling_size
        self.pool_stride = pooling_stride

    def forward(self, x):
        x = m.get_conv2d(x,
            self.W, self.b,
            stride=self.stride, padding=self.padding
            pool_size=self.pool_size, pool_stride=self.pool_stride)