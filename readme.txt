NAS实现协同过滤
search space：
	linear layer: out feature, initial type
	dropout layer: dropout rate
	length
loss function: BCEWithLogitsLoss
迭代优化算法：Adam
fitness: ndcg
cossover: sbx(simulated binary crossover)
mutation: pm(多项式变异)
	  add a new element
	  modify the element
	  delete the element
select: tournament selection

neural network architecture:
	1`embed_user
	2`embed_item 
	 通过torch.cat拼接两个embedding layer的输出，作为dnn的输入
	3`dnn layer
	 通过编码构建DNN结构：full connection unit和dropout unit交替
		full connection unit: linear + ReLu()
		dropout unit: dropout
	4`predict layer:Linear(factor_num * 2, 1)

待解决问题：
1.colab最长连接时常为12小时，修改population size为16，未测试是否能运行完
（google colab杀熟，用的越多，分的越少，辣鸡）
2.选择时仅选择ndcg作为metric，后续是否需要添加其他metrics，添加后如何更改选择条件