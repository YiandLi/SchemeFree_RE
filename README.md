# SchemeFree_RE

调试模型 / debug 阅读：建议使用 `train.py` \
训练脚本使用 `run.sh`->`lightning_runner.py`

TODO:
修改模型为 pre-trained encoder-decoder model



# 方案一
直接使用单个的 span/relation embedding 送入 gpt2 
debug 后跑通，recall 很高，precision 很低


![企业微信截图_16847473297641](https://github.com/YiandLi/SchemeFree_RE/assets/72687714/7ccfbeff-86fe-4b40-97f6-ec4ce0966559)


precision:0.214, recall:0.949, f1_score:0.349


# 方案二
尝试拼接原始句子 + `[ent/rel sep]` + `[representation vector]` ，这里：
1. `[ent/rel sep]` 初始化两个可训练 token
2. 直接拼接，但是需要更新 encoder_attention_mask，-> `concat(encoder_mask, [1,1])`

发现对于同一个句子，模型输出全都一致，比如：
```text
pred: ['was a crew member of']  label: ['occupation']
pred: ['was a crew member of']  label: ['was a crew member of']
pred: ['was a crew member of']  label: ['birthplace']
```

方案一不会出现这样的情况，因为输入完全并行
