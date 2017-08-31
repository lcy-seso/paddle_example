- `paddle.layer.seq_slice` 需要 3 个input
    1. input1：必须是一个序列;
    2. start indices；如果从 data layer feed 数据，类型必须是 `dense_vector`
    3. end indices：如果从data layer feed 数据，类型必须是 `dense_vector`
    4. start indices 和 end indices 两者必须有一个不为 `None` （否则退化为 `indentity projection` 不予考虑）：
        - 如果 start indices 为 `None` 且 end indices 不为 None，会从句子起始位置切到指定的结束位置；
        - 如果 end indices 为 `None` 且 start indices 不为 None，会从指定的起始位置切到句子的结束位置；
    5. 会切出： start<= index <=end 这样一段子序列作为输出序列。当 start = end 时，会输出一个只有一个时间步的序列。
    6. `seq_slice` 层的输出依然是一个序列，输出batch中序列的信息（序列在一个batch中的起始位置）会重新计算。
    7. start 和 end indices 从 0 开始；


- 关于 start indices 和 end indices 参数：
    - 尽管切序列时指定的“起始”和“结束”序号是整形，目前代码实现时把这两者放在 Paddle内部的 `Matrix`中。
    - 这个`Matrix`的数据位只能存放 real 类型，所以通过 data layer feed 数据给 `paddle.layer.seq_slice` 的第二个输入（起始序号）和第三个输入（结束序列）时，请务必使用 `dense_vector` 而不是 `integer_value`或是`integer_value_sequence`系列类型。
    - starts 和 ends 可以含有超过一个序号，比如下面的例子：

    ```python
    输入：
    sequence :
    [0. 0. 0.] [1.0 1.0 1.0] [2.0 2.0 2.0] [3.0 3.0 3.0] [4.0 4.0 4.0]

    starts (paddle.layer.data 指定 type=paddle.data_type.dense_vector_sequence(3)):
    [1.0 3.0 4.0]

    ends (paddle.layer.data 指定 type=paddle.data_type.dense_vector_sequence(3)):
    [2.0 4.0 4.0]

    输出下面的三个子序列：
    [1.0 1.0 1.0] [2.0 2.0 2.0]
    [3.0 3.0 3.0] [4.0 4.0 4.0]
    [4.0 4.0 4.0]
    ```

- starts indices 和 end indices 通过`paddle.layer.data`获取输入数据，必须有固定的维度，但一个 batch 含有多个序列，想切的子序列数目不一样。
    1. 需要为 start indices 和 end indices 对应的`paddle.layer.data`设置一个最大维度；
    2. 比如这个维度设置为5， 但是某一个样本只打算切两段，也就是只提供两个起始位置，只需要在输入向量 2 ~ 4 维全部填充 -1 （必须是-1）

    ```python
    输入的一个batch含有2个序列：
    [0. 0. 0.] [1.0 1.0 1.0] [2.0 2.0 2.0] [3.0 3.0 3.0] [4.0 4.0 4.0]
    [0. 0. 0.] [1.0 1.0 1.0] [2.0 2.0 2.0] [3.0 3.0 3.0] [4.0 4.0 4.0] [5.0 5.0 5.0]

    starts (paddle.layer.data 指定 type=paddle.data_type.dense_vector_sequence(5)):
    [1.0 3.0 -1. -1. -1.]
    [2.0 -1. -1. -1. -1.]

    ends (paddle.layer.data 指定 type=paddle.data_type.dense_vector_sequence(5)):
    [2.0 4.0 -1. -1. -1.]
    [4.0 -1. -1. -1. -1.]

    输出下面的三个子序列：
    [1.0 1.0 1.0] [2.0 2.0 2.0]
    [3.0 3.0 3.0] [4.0 4.0 4.0]
    [3.0 3.0 3.0] [4.0 4.0 4.0] [5.0 5.0 5.0]
    ```
