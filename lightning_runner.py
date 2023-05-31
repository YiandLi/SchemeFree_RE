import sys

sys.path.append("./")
import logging
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertModel, GPT2Config, GPT2LMHeadModel
from Metric import Metric
from data_utils import read_and_load_data, MyDataset
from model import ScFreeModel, CustomGPT2LMHeadModel
from pl_model import LitModel
from train import get_decoder_tokenizer
import os

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(config):
    og_path = hydra.utils.get_original_cwd()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"========== hydra config =========")
    for k, v in config.items():
        logger.info(f"{k}:{v}")
    
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
    ent_decoder = CustomGPT2LMHeadModel(ent_decoder_config)
    ent_decoder.resize_token_embeddings(len(ent_tokenizer.vocab))  # resize output embedding
    
    rel_decoder_config = GPT2Config(n_layer=3, n_positions=128,
                                    bos_token_id=rel_tokenizer.bos_token_id,
                                    pad_token_id=rel_tokenizer.eos_token_id,
                                    add_cross_attention=True)  # 将 encoder 结果作为 memory key
    rel_decoder = CustomGPT2LMHeadModel(rel_decoder_config)
    rel_decoder.resize_token_embeddings(len(rel_tokenizer.vocab))  # resize output embedding
    
    base_model = ScFreeModel(config, encoder, ent_decoder, rel_decoder)
    
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
    
    # TODO: lightning module
    logger.info(
        f"batch_size is {config['batch_size']} , train batch num: {len(train_dataloader)}, valid batch num: {len(dev_dataloader)}")
    
    #     def __init__(self, model, metric, ent_tokenizer, rel_tokenizer, config):
    metric = Metric()
    model = LitModel(base_model, metric, ent_tokenizer, rel_tokenizer, config)
    
    checkpoint_callback = ModelCheckpoint(monitor="eval_epoch_relation_f1", mode='max', save_top_k=1,
                                          save_weights_only=True
                                          )
    earlystop_callback = EarlyStopping(monitor="eval_epoch_relation_f1", patience=5, verbose=False, mode="max")
    
    trainer = pl.Trainer(callbacks=[checkpoint_callback, earlystop_callback],
                         max_epochs=config["epochs"],
                         check_val_every_n_epoch=1,
                         accelerator="auto",
                         num_sanity_val_steps=0,  # for the bug: https://github.com/mindslab-ai/faceshifter/issues/5
                         logger=False,
                         enable_progress_bar=False
                         # 禁用 tqdm https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning
                         )
    
    logging.info("Get pytorch_lightning trainer")
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=dev_dataloader,
                )


if __name__ == '__main__':
    main()
