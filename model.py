import torch
from torch import nn
from transformers import GPT2LMHeadModel


class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    # 为了实现 model.generate( encoder_hidden_states )
    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, past=None, **kwargs):
        res = super().prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        res["encoder_hidden_states"] = kwargs["encoder_hidden_states"]
        return res


class ScFreeModel(nn.Module):
    def __init__(self, args, encoder, ent_decoder, rel_decoder):
        super(ScFreeModel, self).__init__()
        self.encoder = encoder
        self.args = args
        self.rel_decoder = rel_decoder
        self.ent_decoder = ent_decoder
    
    def get_loss(self, all_input_ids, all_input_mask, all_segment_ids,
                 ent_labels=None, rel_labels=None, ent_inputs=None, rel_inputs=None, ent_pos=None, rel_pos=None
                 ):
        assert ent_pos != None, f"[Model] entity position is not provided"
        assert rel_pos != None, f"[Model] relation position is not provided"
        
        # device = all_input_ids.device
        
        batch_outputs = self.encoder(all_input_ids, all_input_mask, all_segment_ids)
        last_hidden_state = batch_outputs.last_hidden_state
        
        # TODO：得到所有实体向量和对应 id
        vector_dict = self.get_ents_and_rels(all_input_ids, last_hidden_state, ent_pos, rel_pos)
        
        # TODO：计算损失
        ent_vectors = vector_dict["gold_ent_vector"][:, None, :]  # batch_size * ent_num , 1, hidden_state
        assert len(ent_vectors) == len(ent_labels), "[Model] |encoder vectors| != |encoder labels|"
        ent_loss = self.ent_decoder(encoder_hidden_states=ent_vectors, input_ids=ent_inputs, labels=ent_labels).loss
        
        rel_vectors = vector_dict["gold_rel_vector"][:, None, :]
        assert len(rel_vectors) == len(rel_labels)
        rel_loss = self.rel_decoder(encoder_hidden_states=rel_vectors, input_ids=rel_inputs, labels=rel_labels).loss
        return ent_loss, rel_loss
    
    def inference(self, all_input_ids, all_input_mask, all_segment_ids,
                  ent_pos, rel_pos, ent_bos_id, rel_bos_id):
        inference_preds = {
            "gold_ent_preds": None,
            "other_ent_preds": None,
            "gold_rel_preds": None,
            "other_rel_preds": None
        }
        
        batch_outputs = self.encoder(all_input_ids, all_input_mask, all_segment_ids)
        last_hidden_state = batch_outputs.last_hidden_state
        vector_dict = self.get_ents_and_rels(all_input_ids, last_hidden_state, ent_pos, rel_pos, mode="inference")
        
        inference_preds["gold_ent_preds"] = self.ent_decoder.generate(
            encoder_hidden_states=vector_dict["gold_ent_vector"][:, None, :],
            inputs=torch.full((len(vector_dict["gold_ent_vector"]), 1), ent_bos_id).type_as(all_input_ids)
        )
        inference_preds["gold_rel_preds"] = self.rel_decoder.generate(
            encoder_hidden_states=vector_dict["gold_rel_vector"][:, None, :],
            inputs=torch.full((len(vector_dict["gold_rel_vector"]), 1), rel_bos_id).type_as(all_input_ids)
        
        )
        inference_preds["other_rel_preds"] = self.rel_decoder.generate(
            encoder_hidden_states=vector_dict["other_rel_vector"][:, None, :],
            inputs=torch.full((len(vector_dict["other_rel_vector"]), 1), rel_bos_id).type_as(all_input_ids)
        )
        
        return inference_preds
    
    def get_ents_and_rels(self, all_input_ids, last_hidden_state,
                          ent_pos_list, rel_pos_list, mode="train"):
        """
        得到所有合法的实体 vector 和对应的 golden_entity_id
        """
        return_dict = {
            "gold_ent_vector": None,
            "other_ent_vector": None,
            "gold_rel_vector": None,
            "other_rel_vector": None
        }
        
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        raw_extend = last_hidden_state.unsqueeze(2).expand(-1, -1, seq_len, -1)
        col_extend = last_hidden_state.unsqueeze(1).expand(-1, seq_len, -1, -1)
        span_vector = torch.div(raw_extend + col_extend, 2)
        
        # TODO: gold_ent_vector -> train
        gold_entity_vector = torch.stack(
            [span_vector[i, j, k] for i, span_list in enumerate(ent_pos_list) for j, k in span_list])
        return_dict["gold_ent_vector"] = gold_entity_vector
        
        # TODO：other_ent_vector (_mask for valid filter) -> eval & test
        if mode != "train":
            # 构造上三角 mask
            _mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=0)
            # padding mask: cls and sep
            valid_token_len = [len(torch.where(i != 0)[0]) for i in all_input_ids]
            
            for i, (single_mask, _token_len, span_list) in enumerate(zip(_mask, valid_token_len, ent_pos_list)):
                single_mask[0] = 0
                single_mask[:, 0] = 0
                single_mask[_token_len - 1:] = 0  # [sep]
                single_mask[:, _token_len - 1:] = 0
                for j, k in span_list:
                    single_mask[j, k] = 0
            
            _mask = _mask.reshape(-1)
            _other_vector = span_vector.reshape(-1, hidden_size)
            _other_vector = _other_vector[_mask == 1]
            
            return_dict["other_ent_vector"] = _other_vector
        
        # TODO: gold_rel_vector & other_rel_vector , all defined by golden entities
        gold_rel_vector = torch.empty((0, hidden_size)).type_as(all_input_ids)  # instance level
        other_rel_vector = torch.empty((0, hidden_size)).type_as(all_input_ids)
        for i, (ent_pos, rel_pos) in enumerate(zip(ent_pos_list, rel_pos_list)):
            # get rel representations
            ent_vectors = torch.stack([span_vector[i, j, k] for j, k in ent_pos])
            raw_ent_vectors = ent_vectors.unsqueeze(1).expand(-1, len(ent_pos), -1)
            col_ent_vectors = ent_vectors.unsqueeze(0).expand(len(ent_pos), -1, -1)
            rel_vector = torch.div(raw_ent_vectors + col_ent_vectors, 2)  # ent_num, ent_num, hidden_state
            
            _mask = torch.zeros(*rel_vector.shape[:-1])  # ent_num, ent_num
            for s_id, o_id in rel_pos:
                gold_rel_vector = torch.vstack((gold_rel_vector, rel_vector[s_id, o_id]))
                _mask[s_id, o_id] = 1
            other_rel_vector = torch.vstack(
                (other_rel_vector, rel_vector.reshape(-1, hidden_size)[0 == _mask.reshape(-1)]))
        return_dict["gold_rel_vector"] = gold_rel_vector
        return_dict["other_rel_vector"] = other_rel_vector
        
        return return_dict
