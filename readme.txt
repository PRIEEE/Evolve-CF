NAS实现协同过滤
search space：
	linear layer: out feature, initial type
	dropout layer: dropout rate
	length
loss function: BCEWithLogitsLoss
迭代优化算法：Adam
fitness: ndcg
learning rate：选择个体时：0.002
	       final train：0.001
cossover: sbx(simulated binary crossover) rate：0.9
mutation: pm(多项式变异) rate：0.2
	  add a new element
	  modify the element
	  delete the element
select: tournament selection（Slack Binary Tournament Selection）
	elitism rate：0.2

neural network architecture:
	1`embed_user
	2`embed_item 
	 通过torch.cat拼接两个embedding layer的输出，作为dnn的输入
	3`dnn layer
	 通过编码构建DNN结构：full connection unit和dropout unit交替
		full connection unit: linear + ReLu()
		dropout unit: dropout
	4`predict layer:Linear(factor_num, 1)

待解决问题：
目前种群规模：16，进化10代
训练结果HR非常优秀，NDCG很一般，说明可以找到用户偏好的item，但是排序不佳
老师给出的改进方案：训练20代

修改种群规模：30，进化20代，未完成
