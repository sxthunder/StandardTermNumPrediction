# StandardTermNumPrediction
针对中文临床术语标准化中标准术语个数预测问题

partition_v1.py partition_v2.py partition_v3.py 是预分段模块，分别代表不同的融合方式（与、或、软融合）
partition_cla.py 生成模块
partition_rank.py 排序模块
partition_cla_rank.py 生成排序联合训练，但效果不佳，最后废弃

可能回看到代码里有fnn_adapter之类的东西，就把它当普通BERT使用
