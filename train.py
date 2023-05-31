import sys

sys.path.append("./")

import os
import torch
import logging
import hydra
import json
from transformers import BertTokenizerFast, BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader

from Metric import Metric
from model import ScFreeModel

from data_utils import read_and_load_data, MyDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_decoder_tokenizer(og_path, args, mode):  # mode: ent / rel
    tokenizer_path = os.path.join(og_path, args.dataset_path, f"{mode}_decoder_tokenizer")
    if os.path.exists(tokenizer_path):
        logging.info(f"load tokenizer from {tokenizer_path}")
        # get decoder_tokenizer
        decoder_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)  # 这里不能用 BertTokenizerFast
        decoder_tokenizer.mask_token, decoder_tokenizer.cls_token = None, None
        
        return decoder_tokenizer
    
    init_decoder_tokenizer = BertTokenizer.from_pretrained(args.encoder_type)
    # 得到预测输出词表
    reader = open(os.path.join(og_path, args.dataset_path, f"{mode}2id.json"), "r")
    ent_labels = list(json.load(reader).keys())
    ent_tokens = set()
    vocab_len = 0
    
    for i in ent_labels:
        label_ids = init_decoder_tokenizer.tokenize(i)
        ent_tokens.update(label_ids)
        vocab_len += (1 + len(label_ids))
    vocab_len += 1
    
    print(f"The ideal max seq length w/o repetition is {vocab_len}")
    
    # 生成，覆盖词表
    init_decoder_tokenizer.eos_token = '[EOS]'
    init_decoder_tokenizer.bos_token = '[CLS]'
    init_decoder_tokenizer.mask_token = None
    init_decoder_tokenizer.pad_token = '[EOS]'
    ent_tokens.update(['[CLS]', '[SEP]', '[EOS]'] + init_decoder_tokenizer.all_special_tokens)
    
    init_decoder_tokenizer.save_pretrained(tokenizer_path)
    open(os.path.join(tokenizer_path, "vocab.txt"), "w").write("\n".join(ent_tokens))
    
    # get decoder_tokenizer
    decoder_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)  # 这里不能用 BertTokenizerFast
    decoder_tokenizer.mask_token, decoder_tokenizer.cls_token = None, None
    
    return decoder_tokenizer


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(config):
    og_path = hydra.utils.get_original_cwd()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"========== device: {device} =========")
    if torch.cuda.is_available():
        logger.info("=" * 20 + f"Using Gpu: {torch.cuda.get_device_name(0)} " + "=" * 20)

    
    torch.manual_seed(42)  # pytorch random seed
    torch.backends.cudnn.deterministic = True
    
    # TODO: get tokenizer
    encoder_tokenizer = BertTokenizerFast.from_pretrained(config.encoder_type)
    ent_tokenizer = get_decoder_tokenizer(og_path, config, "ent")
    rel_tokenizer = get_decoder_tokenizer(og_path, config, "rel")
    
    # TODO: get model
    encoder = BertModel.from_pretrained(config.encoder_type)
    ent_decoder_config = GPT2Config(n_layer=3, n_positions=128,
                                    bos_token_id=ent_tokenizer.bos_token_id,
                                    pad_token_id=ent_tokenizer.eos_token_id,
                                    add_cross_attention=True)  # 将 encoder 结果作为 memory key
    ent_decoder = GPT2LMHeadModel(ent_decoder_config)
    ent_decoder.resize_token_embeddings(len(ent_tokenizer.vocab))  # resize output embedding
    
    rel_decoder_config = GPT2Config(n_layer=3, n_positions=128,
                                    bos_token_id=rel_tokenizer.bos_token_id,
                                    pad_token_id=rel_tokenizer.eos_token_id,
                                    add_cross_attention=True)  # 将 encoder 结果作为 memory key
    rel_decoder = GPT2LMHeadModel(rel_decoder_config)
    rel_decoder.resize_token_embeddings(len(rel_tokenizer.vocab))  # resize output embedding
    
    model = ScFreeModel(config, encoder, ent_decoder, rel_decoder)
    
    # TODO: get data
    train_data = read_and_load_data(og_path, config, "train", encoder_tokenizer,
                                    ent_tokenizer, rel_tokenizer)
    dev_data = read_and_load_data(og_path, config, "valid", encoder_tokenizer,
                                  ent_tokenizer, rel_tokenizer)
    train_dataset = MyDataset(train_data, config, encoder_tokenizer, ent_tokenizer, rel_tokenizer)
    dev_dataset = MyDataset(dev_data, config, encoder_tokenizer, ent_tokenizer, rel_tokenizer)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=6 if device == 'cuda' else 0,
                                  drop_last=False,
                                  collate_fn=train_dataset.generate_batch,
                                  )
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size * 2,
                                shuffle=False,
                                num_workers=6 if device == 'cuda' else 0,
                                drop_last=False,
                                collate_fn=dev_dataset.generate_batch,
                                )
    
    # for batch in train_dataloader:
    #     for t in batch[:7]: t.to(device)
    #     loss = model.get_loss(*batch)
    
    metric = Metric()
    for batch in dev_dataloader:
        for t in batch[:7]: t.to(device)
        pred_dict = model.inference(*batch[:3], *batch[7:], ent_tokenizer.bos_token_id, rel_tokenizer.bos_token_id)
        ent_labels, rel_labels = batch[3:5]
        metric.update_eval(pred_dict, ent_labels, rel_labels, ent_tokenizer, rel_tokenizer)
        # ent_metric.update_entity_eval(ent_preds, ent_labels, ent_tokenizer)
    
    metric.get_result()


if __name__ == '__main__':
    main()
