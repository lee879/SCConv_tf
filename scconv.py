import tensorflow as tf
from tensorflow.python.keras.layers import Layer,Activation,Conv2D,GlobalAveragePooling2D
from tensorflow.python.keras import Sequential,Model

class GroupNormalization_sp(Layer):
    def __init__(self,
                 num_groups: int,
                 num_channels: int,
                 eps: float = 1e-6,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 **kwargs
                 ):
        super(GroupNormalization_sp, self).__init__(**kwargs)
        assert num_channels % num_groups == 0, "the num_channels is invalid"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=(self.num_channels,),
                                     initializer=self.gamma_initializer,
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=(self.num_channels,),
                                    initializer=self.beta_initializer,
                                    trainable=True)
        super(GroupNormalization_sp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)
        group_size = input_shape[-1] // self.num_groups
        grouped_inputs = tf.reshape(inputs, [-1, self.num_groups, group_size])

        group_mean = tf.reduce_mean(grouped_inputs, axis=[-1], keepdims=True)
        group_var = tf.reduce_mean(tf.square(grouped_inputs - group_mean), axis=[-1], keepdims=True)

        normalized_inputs = (grouped_inputs - group_mean) / tf.sqrt(group_var + self.eps)

        normalized_inputs = tf.reshape(normalized_inputs, input_shape)

        normalized_output = self.gamma * normalized_inputs + self.beta

        normalized_w = self.gamma / tf.reduce_sum(self.gamma)

        output = normalized_output * normalized_w

        return output

class SRU(Layer):
    def __init__(self, num_channels: int, num_groups: int, ratio: int = 2):
        super(SRU, self).__init__()
        self.gn = GroupNormalization_sp(num_groups=num_groups, num_channels=num_channels)
        self.ac = Activation(tf.nn.sigmoid)
        self.ratio = ratio
        self.num_channels = num_channels

    def batch_median(self,tensor):
        # 获取批次数目
        batch_size = tf.shape(tensor)[0]
        # 将张量展平，保留最后一维
        flattened_tensor = tf.reshape(tensor, [batch_size, -1, tf.shape(tensor)[-1]])
        # 计算每个批次的中位数索引
        k = tf.shape(flattened_tensor)[-2] // 2  # 中位数在排序后的位置索引（注意这里取整除法）
        indices = tf.range(batch_size) * tf.shape(flattened_tensor)[1] + k
        # 找到每个批次的中位数值
        median_values = tf.gather(tf.reshape(tf.sort(flattened_tensor), [-1, tf.shape(tensor)[-1]]), indices)

        return median_values
    def call(self, inputs, **kwargs):
        normal_inputs = self.ac(self.gn(inputs))

        threshold = tf.reduce_mean(self.batch_median(normal_inputs))

        x1 = tf.cast(normal_inputs > threshold, tf.float32)
        x2 = tf.cast(normal_inputs <= threshold, tf.float32)

        y1 = inputs * x1
        y2 = inputs * x2

        z0_list = tf.split(y1, num_or_size_splits=[self.num_channels // self.ratio, self.num_channels // self.ratio], axis=-1)
        z1_list = tf.split(y2, num_or_size_splits=[self.num_channels // self.ratio, self.num_channels // self.ratio], axis=-1)

        h1 = tf.add(z0_list[0], z1_list[1])
        h2 = tf.add(z1_list[0], z0_list[1])

        out = tf.concat([h1, h2], axis=-1)
        return out

class CRU(Layer):
    def __init__(self, input_channels: int, ratio: float = 0.5, kernel_size: int = 3, num_groups: int = 2):
        super(CRU, self).__init__()
        assert ratio == 0.5, "The ratio must be 0.5"
        self.input_channels = input_channels
        self.high_channels = tf.cast(self.input_channels * ratio, tf.int32)
        self.low_channels = tf.cast(self.input_channels - self.high_channels, tf.int32)
        self.point_conv_1 = Conv2D(self.high_channels, 1, 1, "same")
        self.point_conv_2 = Conv2D(self.low_channels, 1, 1, "same")

        self.GWC_high = Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=1, groups=num_groups, padding="same"),
            Conv2D(input_channels, 1, 1, "same")
        ])
        self.PWC_high = Conv2D(input_channels, 1, 1, "same")
        self.PWC_low = Conv2D(self.low_channels, 1, 1, "same")
        self.gap = GlobalAveragePooling2D()
        self.ac = Activation(tf.nn.softmax)

    def call(self, inputs, **kwargs):
        input_list = tf.split(inputs, num_or_size_splits=[self.high_channels, self.low_channels], axis=-1)
        y0 = self.point_conv_1(input_list[0])
        y1 = self.point_conv_2(input_list[1])

        y0_0 = self.GWC_high(y0)
        y0_1 = self.PWC_high(y0)

        y1_0 = self.PWC_low(y1)

        z0 = tf.add(y0_0, y0_1)
        z1 = tf.concat([y1, y1_0], axis=-1)

        h0 = self.gap(z0)
        h1 = self.gap(z1)

        h2 = self.ac(tf.concat([h0, h1], axis=-1))

        output_0 = tf.expand_dims(tf.expand_dims(h2[:, :self.input_channels], axis=1), axis=1) * z0
        output_1 = tf.expand_dims(tf.expand_dims(h2[:, self.input_channels:], axis=1), axis=1) * z1

        out = tf.add(output_0, output_1)
        return out

class SCConv(Model):
    def __init__(self, input_channels, num_groups=8):
        super(SCConv, self).__init__()
        self.sru = SRU(num_channels=input_channels, num_groups=num_groups)
        self.cru = CRU(input_channels=input_channels)
        self.point_conv_input = Conv2D(input_channels, 1, 1, "same")
        self.point_conv_out = Conv2D(input_channels, 1, 1, "same")

    def call(self, inputs, training=None, mask=None):
        x = self.point_conv_input(inputs)
        y0 = self.sru(x)
        y1 = self.cru(y0)
        z = self.point_conv_out(y1)
        out = tf.add(inputs, z)
        return out


m = SCConv(input_channels=16)
x = tf.random.normal(shape=(4,64,64,16))
y = m(x)
print(y.shape)





