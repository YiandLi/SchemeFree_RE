import os, logging, json, torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, text, ent_infos, rel_infos):
        self.text = text
        self.ent_infos = ent_infos
        self.rel_infos = rel_infos


def read_and_load_data(og_path, args, mode, encoder_tokenizer, ent_tokenizer, rel_tokenizer):
    # if os.path.exists(f"cache/{mode}_{args.way_num}_{args.shot_num}.pth"):
    #     return torch.load(f"cache/{mode}_{args.way_num}_{args.shot_num}.pth")
    
    data = json.load(open(os.path.join(og_path, args.dataset_path, f"{mode}_data.json"), "r"))
    # data = json.load(open(os.path.join(og_path, args.dataset_path, f"{mode}_data.json"), "r"))[:20]  # for test
    
    logger.info("Reading tasks from {}...".format(os.path.join(args.dataset_path, f"{mode}_data.json")))
    logger.info(
        f"For decoder output, the ent label and rel label separator are {ent_tokenizer.sep_token}({ent_tokenizer.sep_token_id}) "
        f"and {rel_tokenizer.sep_token}({rel_tokenizer.sep_token_id})")
    
    input_features = []
    for instance_dict in tqdm(data, desc=f"load {mode} data"):
        fea = convert_example_to_features(instance_dict, encoder_tokenizer, args, ent_tokenizer, rel_tokenizer)
        if fea: input_features.append(fea)  # 越界情况跳过
    
    logger.info(f"Get {len(input_features)} instances from file : {mode}_data.json")
    return input_features


def convert_example_to_features(
        instance_dict, tokenizer, args, ent_tokenizer, rel_tokenizer
):
    ent_return, rel_return = defaultdict(set), defaultdict(set)
    
    token2char_span_mapping = \
        tokenizer(instance_dict['text'], return_offsets_mapping=True, max_length=args.max_seq_len, truncation=True)[
            "offset_mapping"]
    
    if not token2char_span_mapping:  # ! 有特殊字符报错("\u2063") 导致 word_tokens = []
        return None
    
    # { 每个token的开始字符的索引: 第几个token } and { end_index : token_index }
    start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
    end_mapping = {j[-1]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
    
    # 将raw_text的下标 与 token的start和end下标对应
    for ent_info in instance_dict["entity_list"]:  # 全都是index索引，label不用额外转换
        [start, end], type = ent_info['char_span'], ent_info['type']
        
        if start in start_mapping and end in end_mapping:
            # 得到概率矩阵中的 横竖索引
            # 其实就是span 的 start_idx 和 end_idx 控制的位置
            start = start_mapping[start]
            end = end_mapping[end]
            ent_return[(start, end)].add(type)
        else:
            print(f"\tEntity {ent_info['char_span']} out of max seq_len {args.max_seq_len}, "
                  f"text {instance_dict['text'][:50]} ...")
    
    for rel_info in instance_dict['relation_list']:
        sub_start, sub_end = rel_info['subj_char_span']
        obj_start, obj_end = rel_info['obj_char_span']
        
        if sub_start in start_mapping and sub_end in end_mapping and obj_start in start_mapping and obj_end in end_mapping:
            sub_start, sub_end = start_mapping[sub_start], end_mapping[sub_end]
            obj_start, obj_end = start_mapping[obj_start], end_mapping[obj_end]
            type = rel_info['predicate']
            rel_return[(sub_start, sub_end, obj_start, obj_end)].add(type)
        else:
            print(
                f"\tRelation ({rel_info['subject']}, {rel_info['predicate']}, {rel_info['object']}) out of max seq_len {args.max_seq_len}, "
                f"text {instance_dict['text'][:50]} ...")
    
    ent_bos, ent_sep, ent_eos = ent_tokenizer.bos_token, ent_tokenizer.sep_token, ent_tokenizer.eos_token
    rel_bos, rel_sep, rel_eos = rel_tokenizer.bos_token, rel_tokenizer.sep_token, rel_tokenizer.eos_token
    
    if ent_return and rel_return:  # 直接整理成为 decoder 输出格式 先不加 cls， eos
        for i in ent_return: ent_return[i] = f"{ent_bos} " + f" {ent_sep} ".join(ent_return[i]) + f" {ent_eos}"
        for i in rel_return: rel_return[i] = f"{rel_bos} " + f" {rel_sep} ".join(rel_return[i]) + f" {ent_eos}"
        return InputFeatures(instance_dict['text'], ent_return, rel_return)
    else:
        return None


class MyDataset(Dataset):
    def __init__(self, data, args, encoder_tokenizer, ent_tokenizer, rel_tokenizer):
        self.data = data
        self.args = args
        self.encoder_tokenizer = encoder_tokenizer
        self.ent_tokenizer = ent_tokenizer
        self.rel_tokenizer = rel_tokenizer
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        seq_dims : 控制维度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs],
                            axis=0)  # label_num, max_seq_len, max_seq_len，注意这里 max_seq_len 是同batch内最长句子的长度
        elif not hasattr(length, '__getitem__'):
            length = [length]
        
        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        
        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            
            # pad_width是在各维度的各个方向上想要填补的长度,如（（1，2），（2，2））
            # 表示在第一个维度上水平方向上padding=1,垂直方向上padding=2
            # 在第二个维度上水平方向上padding=2,垂直方向上padding=2。
            # 如果直接输入一个整数，则说明各个维度和各个方向所填补的长度都一样。
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)
        
        return np.array(outputs)
    
    def generate_batch(self, features):
        """
            inputs = {"input_ids": torch.tensor([input_ids])}
            outputs = model(**inputs, labels=torch.tensor([input_ids]))
        """
        batch_input_ids, batch_input_mask, batch_segment_ids = [], [], []  # for encoder
        ent_pos, rel_pos, ent_labels, rel_labels = [], [], [], []  # for decoder
        
        for idx, _feature in enumerate(features):
            encoder_txt = self.encoder_tokenizer.encode_plus(_feature.text, max_length=self.args.max_seq_len,
                                                             truncation=True)
            batch_input_ids.append(encoder_txt["input_ids"])
            batch_input_mask.append(encoder_txt["token_type_ids"])
            batch_segment_ids.append(encoder_txt["attention_mask"])
            
            ent_pos_tuples = list(_feature.ent_infos.keys())
            ent_pos_tuples2id = {k: i for i, k in enumerate(ent_pos_tuples)}
            
            ent_pos.append(ent_pos_tuples)  # instance level
            rel_pos.append([(ent_pos_tuples2id[(s_s, s_e)], ent_pos_tuples2id[(o_s, o_e)]) for s_s, s_e, o_s, o_e \
                            in list(_feature.rel_infos.keys())])
            # ent_pos.extend([(idx, *j) for j in list(_feature.ent_infos.keys())])
            # rel_pos.extend([(idx, *j) for j in list(_feature.rel_infos.keys())])
            ent_labels.extend(list(_feature.ent_infos.values()))  # 一个句子一个元素，每个句子元素包含所有实体的顺序标签
            rel_labels.extend(list(_feature.rel_infos.values()))
        
        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask)).float()
        batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        
        ent_labels = self.ent_tokenizer.batch_encode_plus(ent_labels, add_special_tokens=False).input_ids
        rel_labels = self.rel_tokenizer.batch_encode_plus(rel_labels, add_special_tokens=False).input_ids
        
        # padding
        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask)).float()
        
        # gpt2 不计算 -100 的损失
        ent_inputs = torch.tensor(self.sequence_padding(ent_labels, value=self.ent_tokenizer.sep_token_id))
        rel_inputs = torch.tensor(self.sequence_padding(rel_labels, value=self.rel_tokenizer.sep_token_id))
        
        ent_labels = torch.tensor(self.sequence_padding(ent_labels, value=-100))
        rel_labels = torch.tensor(self.sequence_padding(rel_labels, value=-100))
        
        return batch_input_ids, batch_input_mask, batch_segment_ids, \
               ent_labels, rel_labels, ent_inputs, rel_inputs, ent_pos, rel_pos
