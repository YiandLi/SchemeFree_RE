import pytorch_lightning as pl
import logging

import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class LitModel(pl.LightningModule):
    def __init__(self, base_model, metric, ent_tokenizer, rel_tokenizer, config):
        super().__init__()
        
        self.model = base_model
        self.metric = metric
        self.ent_tokenizer = ent_tokenizer
        self.rel_tokenizer = rel_tokenizer
        self.config = config
    
    def training_step(self, batch_train_data, batch_idx):  # TODO 定义 train 过程
        ent_loss, rel_loss = self.model.get_loss(*batch_train_data)
        if batch_idx % 250 == 0 or batch_idx < 5:
            logging.info(
                f"\t[epoch {self.current_epoch} - {batch_idx}] entity loss: {ent_loss.item():.3f}, relation loss: {rel_loss.item():.3f} ")
        
        # ??? ent 不需要太怎么训练
        if ent_loss.item() > 0:
            loss = ent_loss + rel_loss
        else:
            loss = rel_loss
        
        return {'loss': loss, 'ent_loss': ent_loss.item(), 'rel_loss': rel_loss.item()}
    
    def training_epoch_end(self, outputs):
        mean_ent_loss = sum(i['ent_loss'] for i in outputs) / len(outputs)
        mean_rel_loss = sum(i['rel_loss'] for i in outputs) / len(outputs)
        logging.info(
            f"[epoch {self.current_epoch}] mean_ent_loss: {mean_ent_loss:.3f}, mean_rel_loss: {mean_rel_loss:.3f}\n"
            # "learning_rate": optimizer.param_groups[0]['lr'],
        )
    
    def validation_step(self, batch_valid_data, batch_idx):  # TODO 定义 eval 过程
        with torch.no_grad():
            pred_dict = self.model.inference(*batch_valid_data[:3], *batch_valid_data[7:],
                                             self.ent_tokenizer.bos_token_id, self.rel_tokenizer.bos_token_id)
        ent_labels, rel_labels = batch_valid_data[3:5]
        self.metric.update_eval(pred_dict, ent_labels, rel_labels, self.ent_tokenizer, self.rel_tokenizer)
        
        # ??? 增加输出部分
        writer = open("eval_cases.txt", "a")
        writer.write("\n".join([f"pred: {p} \nlabel: {l}\n" for p, l in
                                zip(self.rel_tokenizer.batch_decode(pred_dict['gold_rel_preds']),
                                    self.rel_tokenizer.batch_decode(rel_labels))]))
        
        # print(f"valid step {batch_idx}")
    
    def validation_epoch_end(self, outputs) -> None:
        (ent_precision, ent_recall, ent_f1_score), (
            rel_precision, rel_recall, rel_f1_score), log_content = self.metric.get_result()
        logging.info(log_content)
        self.metric.refresh()
        self.log("eval_epoch_relation_f1", rel_f1_score)
    
    def configure_optimizers(self):
        init_learning_rate = float(self.config["lr"])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=init_learning_rate)
        return optimizer
