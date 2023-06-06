# SchemeFree_RE

调试模型 / debug 阅读：建议使用 `train.py` \
训练脚本使用 `run.sh`->`lightning_runner.py`

TODO:
修改模型为 pre-trained encoder-decoder model



## 方案一 v1

![企业微信截图_16847473297641](https://github.com/YiandLi/SchemeFree_RE/assets/72687714/7ccfbeff-86fe-4b40-97f6-ec4ce0966559)

直接使用单个的 span/relation embedding 送入 gpt2 
debug 后跑通，recall 很高，precision 很低：

    precision:0.214, recall:0.949, f1_score:0.349

precision 很低，考虑是因为没有加 True negative span representation 的损失，但是加了之后反而效果下降了：

    precision:0.118, recall:0.522, f1_score:0.192
    
猜测是类别不平均导致 ，这边平均一下

    

## 方案二 v2
尝试拼接原始句子 + `[ent/rel sep]` + `[representation vector]` ，这里：
1. `[ent/rel sep]` 初始化两个可训练 token
2. 直接拼接，但是需要更新 encoder_attention_mask，-> `concat(encoder_mask, [1,1])`


	precision:0.096, recall:0.426, f1_score:0.156



发现对于同一个句子，模型输出全都一致，比如：
```text
pred: ['was a crew member of']  label: ['occupation']
pred: ['was a crew member of']  label: ['was a crew member of']
pred: ['was a crew member of']  label: ['birthplace']
```

方案一不会出现这样的情况，因为输入完全并行

## 方案三 v3
`[cls] [ent / rel vector]`


# 问题

1. 训练时，TN 是否需要进行采样训练，显存会爆炸
    由于已经知道合法span，在训练 entity 时候可以采样几个 non ent，即 `[bos] [eos]` ，二次筛选
    训练 rel 时候可以使用 ent2ent 概率矩阵进行训练
    
2. v2 发现效果不好，可能的原因是区别很小 ，只有一个token，可以考虑：
    1。 直接将 entity 列出来，让 encoder 进行 multi-pass
    2。 先 one-pass，然后后面多列出来几个 token，而不是聚合的 token
