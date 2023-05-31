import datetime


class Metric():
    def __init__(self, output_path=None):
        super().__init__()
        self.output_path = output_path
        self.rel_correct_num, self.rel_predict_num, self.rel_gold_num = 1e-10, 1e-10, 1e-10
        self.ent_correct_num, self.ent_predict_num, self.ent_gold_num = 1e-10, 1e-10, 1e-10
    
    def get_labels(self, _ids, tokenizer):
        _ids = _ids.cpu().numpy().tolist()
        # 先整理 generate/golden token ids ， 删 cls，eos ， 切分 pad
        _ids = [i[1:] for i in _ids]  # delete [cls]
        _ids = [i + [tokenizer.eos_token_id] for i in _ids]  # if [eos] not in list, avoid exception
        _ids = [i[:i.index(tokenizer.eos_token_id)] for i in _ids]  # truncate [eos] + [pad]
        for i, ids in enumerate(_ids): _ids[i] = "".join([str(m) for m in ids])
        _ids = [i.split(str(tokenizer.sep_token_id)) for i in _ids]  # split by [pad]
        _ids = [[i for i in g if i] for g in _ids]  # remove ''  , like case "[sep] [sep]"
        return _ids
    
    def update_eval(self, pred_dict, ent_labels, rel_labels, ent_tokenizer, rel_tokenizer):
        # 处理 entity TP 部分
        ent_pred_gold_ids = self.get_labels(pred_dict['gold_ent_preds'], ent_tokenizer)
        ent_label_ids = self.get_labels(ent_labels, ent_tokenizer)
        for labels, preds in zip(ent_label_ids, ent_pred_gold_ids):
            labels, preds = set(labels), set(preds)
            self.ent_correct_num += len(labels & preds)
            self.ent_gold_num += len(labels)
            self.ent_predict_num += len(preds)
        
        # 处理 entity TN 部分
        if pred_dict['other_ent_preds'] != None:
            ent_pred_other_ids = self.get_labels(pred_dict['other_ent_preds'], ent_tokenizer)
            for negative_pres in ent_pred_other_ids:
                self.ent_predict_num += len(set(negative_pres))
        
        # 处理 relation TP 部分
        rel_pred_gold_ids = self.get_labels(pred_dict['gold_rel_preds'], rel_tokenizer)
        rel_label_ids = self.get_labels(rel_labels, rel_tokenizer)
        for labels, preds in zip(rel_label_ids, rel_pred_gold_ids):
            labels, preds = set(labels), set(preds)
            self.rel_correct_num += len(labels & preds)
            self.rel_gold_num += len(labels)
            self.rel_predict_num += len(preds)
        
        # 处理 relation TN 部分
        if pred_dict['other_rel_preds'] != None:
            rel_pred_other_ids = self.get_labels(pred_dict['other_rel_preds'], ent_tokenizer)
            for negative_pres in rel_pred_other_ids:
                self.rel_predict_num += len(set(negative_pres))
    
    def get_result(self):
        # rel
        rel_precision = self.rel_correct_num / self.rel_predict_num
        rel_recall = self.rel_correct_num / self.rel_gold_num
        rel_f1_score = 2 * rel_precision * rel_recall / (rel_precision + rel_recall)
        
        output = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n' \
                 f'Relation:\n' \
                 f'\tcorrect_num: {self.rel_correct_num:.0f}, predict_num: {self.rel_predict_num:.0f}, gold_num: {self.rel_gold_num:.0f}\n' \
                 f'\tprecision:{rel_precision:.3f}, recall:{rel_recall:.3f}, f1_score:{rel_f1_score:.3f}\n'
        
        # ent
        ent_precision = self.ent_correct_num / self.ent_predict_num
        ent_recall = self.ent_correct_num / self.ent_gold_num
        ent_f1_score = 2 * ent_precision * ent_recall / (ent_precision + ent_recall)
        
        output += f'Entity:\n' \
                  f'\tcorrect_num: {self.ent_correct_num:.0f}, predict_num: {self.ent_predict_num:.0f}, gold_num: {self.ent_gold_num:.0f}\n' \
                  f'\tprecision:{ent_precision:.3f}, recall:{ent_recall:.3f}, f1_score:{ent_f1_score:.3f}\n\n'
        
        # open(self.output_path, "a").write(output)
        # print(output)
        return (ent_precision, ent_recall, ent_f1_score), (rel_precision, rel_recall, rel_f1_score), output
    
    def refresh(self):
        self.rel_correct_num, self.rel_predict_num, self.rel_gold_num = 1e-10, 1e-10, 1e-10
        self.ent_correct_num, self.ent_predict_num, self.ent_gold_num = 1e-10, 1e-10, 1e-10
