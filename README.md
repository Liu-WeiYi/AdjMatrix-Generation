# AdjMatrix-Generation
采用GAN的思想生成任意的拓扑结构
## 划分图的工具——metis
用法1 直接划分
```
./gpmetis <filename> <partition_num>
```
用法2 指定划分出的所有子图均连续
```
./gpmetis -config <filename> <partition_num>
```