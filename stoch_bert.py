import torch
import torch.nn as nn

from packaging import version

from transformers import DistilBertForSequenceClassification, DistilBertModel, AdamW, BertConfig
from transformers import BertForSequenceClassification, BertModel, AdamW, BertConfig

from transformers.models.distilbert.modeling_distilbert import Transformer, Embeddings, TransformerBlock, MultiHeadSelfAttention, FFN
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers.models.bert.modeling_bert import BertLayer, BertEncoder, BertEmbeddings, BertAttention, BertOutput, BertIntermediate, BertSelfAttention, BertSelfOutput, BertPooler, BaseModelOutputWithPastAndCrossAttentions

from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
import copy
import math

# ratio = 0

class ECNet(nn.Module):
    def __init__(self, inp, hid, out):
        super().__init__()
        self.fc1 = nn.Linear(inp, hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid, out)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

import VAE 

class MyMultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.custom_config = {'miss_ratio': 0}
        self.pruned_heads = set()
        # self.gamma = nn.Parameter(torch.tensor([0.1]))
        # self.gamma = nn.Parameter(torch.tensor([1.0] * 12))

        # Added params
        # self.ecfc1 = torch.nn.Linear(config.dim, config.hidden_dim)
        # self.ecact = nn.ReLU()
        # self.ecfc2 = torch.nn.Linear(config.hidden_dim, 60)
        # self.ecdropout = torch.nn.Dropout(config.dropout)
        self.ecnets = nn.ModuleList([VAE.LinearVAE(in_f=64 * 8,hidden_f=64 * 4, enc_f=32, out_f=64*4) for _ in range(3)])


    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
        

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """

  
        bs, q_length, dim = query.size()
        # ec_code = self.ecfc1(query)
        # ec_code = self.ecact(ec_code)
        # ec_code = self.ecfc2(ec_code)
        # ec_code = self.ecdropout(ec_code)
        # print(query.shape, ec_code.shape)
   

        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        
        ## Churn Simulation
        # miss_ids = torch.rand(12) < self.custom_config['miss_ratio']
        
        ## Group Churn Simulation 
        # print('lol')
        # print(self.custom_config)
        
        # miss_ids = torch.rand(12) < self.custom_config['miss_ratio']
        # miss_ids = miss_ids.repeat_interleave(1)

        miss_ids = torch.ones(12)<-1

        devices = self.custom_config['devices']
        # print(self.custom_config['devices'])
        ind = self.custom_config['distribution_matrix'][self.custom_config['it']]
        # print(ind)
        # 1/0

        # print(miss_ids)
        # print('------------')
        for i in range(len(devices)):
            if devices[i] == False:
                miss_ids[ind==i] = True

        


        # print(miss_ids)

                # print("WTF!")
        ## ToDo: Check unavailability of each of the 3 outputs and use the respective ECNets if necessary
        

        # ec_code = ec_code.reshape(bs, self.n_heads, q_length, -1)
        # print(context.shape, ec_code.shape)

        # 1/0
        # print(miss_ids)

        ## Zero Replacement

        # print([context[:,i*4:i*4+4,:,:].sum().item() for i in range(3)])
        # print(miss_ids)
        # print(self.custom_config)
        # print(self.custom_config['intermediary_results'])

        # self.custom_config['intermediary_results'] 
        context[:,miss_ids,:,:] = 0
        
        # print([context[:,i*4:i*4+4,:,:].sum().item() for i in range(3)])
        

        # for i in range(3):
        #     if miss_ids[i*4]:
        #         # print(i, context.shape)

        #         if i == 0:
        #             inp = context[:,i*4+4:,:,:]
        #         elif i == 2:
        #             inp = context[:,:i*4,:,:]
        #         else:
        #             inp = torch.cat((context[:,:i*4,:,:], context[:,i*4+4:,:,:]), dim=1)
                
        #         # print(inp.shape)
        #         reshaped_inp = inp.transpose(1, 2).contiguous().view(bs, -1, 8 * dim_per_head)
        #         # print(reshaped_inp.shape)
        #         out, mu, logvar = self.ecnets[i](reshaped_inp)
        #         out = out.view(bs, -1, 4, dim_per_head).transpose(1,2)

        #         context[:,i*4:i*4+4,:,:] = out
        #         print(out.shape)


        # print([context[:,i*4:i*4+4,:,:].sum().item() for i in range(3)])
                
        # print(f'id: {self.custom_config["it"]}, sum: {context.sum()}')

        ## Weighted Mean Redundancy
        # print((shape(query)[:,1:,:,:]).shape)
        # tmp = self.gamma[1:].unsqueeze(1).unsqueeze(2).unsqueeze(0)
        # # print(tmp.shape)
        # _, _, token_num, head_emb_len = (shape(query)).shape
        # tmp = tmp.repeat(1, 1, token_num, head_emb_len)
        # # print(tmp.shape)
        # red = (shape(query)[:,1:,:,:] * tmp).sum(dim=1)
        # context[:,0,:,:] = red

        ## Mean Redundancy
        # print((shape(query)[:,1:,:,:]).shape)
        # red = shape(query)[:,1:,:,:].sum(dim=1)
        # context[:,0,:,:] = self.gamma * red

        ## Direct Shortcut
        # miss_ids = miss_ids.unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        # miss_ids = miss_ids.repeat(context.shape[0], 1, context.shape[2], context.shape[3]).float()
        # context = (1-miss_ids) * context + miss_ids * shape(query) * self.gamma
        
        # print(self.gamma)
        # print(context.sum())

        # sm = miss_ids.sum()
        # if sm < 12:
        #     coeff = 12.0 / (12 - sm)
        #     # print(sm, coeff)
        #     context[:,miss_ids==False, :, :] *= coeff
            # print(context.sum())
        # print('-------------')
        # print(context.var())
        # print(context.shape)
        context = unshape(context)  # (bs, q_length, dim)
        # print(miss_ids)

        # for ind, tmp in enumerate(context[0,0,:]):
        #     print(round(tmp.item(),2), end=' ')
        #     if ind % 64 == 0:
        #         print()
        
        # print('-------')
        # for ind, tmp in enumerate(context[0,3,:]):
        #     print(round(tmp.item(), 2), end=' ')
        #     if ind % 64 == 0:
        #         print()
        # print('-------')
        
        # print(context.shape)
        # 1/0
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


    def set_custom_config(self, custom_config):
        self.custom_config = custom_config

class MyTransformerBlock(TransformerBlock):
    def __init__(self, config):
        super(MyTransformerBlock, self).__init__(config)

        assert config.dim % config.n_heads == 0

        self.attention = MyMultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        # print(config)

        # x = torch.nn.Linear(config.dim, 5)
        
    def set_custom_config(self, custom_config):
        self.custom_config = custom_config
        self.attention.set_custom_config(custom_config)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output

class MyTransformer(Transformer):

    def __init__(self, config):
        super(MyTransformer, self).__init__(config)
    
        self.n_layers = config.n_layers
        layer = MyTransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def set_custom_config(self, custom_config):
        self.custom_config = custom_config
        for l in self.layer:
            l.set_custom_config(custom_config)

    def forward(
        self, x, attn_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=None
    ):  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x

        intermediary_res = x.detach().unsqueeze(3)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions
            )
            hidden_state = layer_outputs[-1]
            
            # print(intermediary_res.shape)
            intermediary_res = torch.cat((intermediary_res, hidden_state.detach().unsqueeze(3)), dim=3).detach()

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

            self.custom_config['it'] += 1
            self.set_custom_config(self.custom_config) # to broadcast the new config (including the new iterator)
        
        # print(intermediary_res.shape)
        if 'gather_intermediary_results' in self.custom_config and self.custom_config['gather_intermediary_results']:
            self.custom_config['intermediary_results'].append(intermediary_res)
        # print('\n\n------\n------\n\n')
        # net = self.layer[0].attention.ecnets[0]
        # for p in net.parameters():
        #     print(p.name, p.requires_grad, p.data.sum())
        # print(self.layer[0].attention.ecnets[0].parameters())
        # 1/0
        # Add last layer

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        # 1/0 
        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )

import copy

class MyEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
            )

    def set_custom_config(self, custom_config):
        self.custom_config = custom_config

    def forward(self, input_ids):
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.

        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        seq_length = input_ids.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        # print(self.word_embeddings)
        # 1/0
        # if 'mask_embeddings' in self.custom_config:
        #     tmp = self.custom_config['mask_embeddings']
        #     for token in tmp:
        #         self.word_embeddings[token] = 0

        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings
    

    def update_copy(self):
        self.emb_cp = copy.deepcopy(self.word_embeddings.state_dict())
        
    def deactivate_embeddings(self, tokens):
        with torch.no_grad():
            for token in tokens:
                    # self.word_embeddings.weight[token].requires_grad = False
                self.word_embeddings.weight[token] = 0
            
    def activate_all_embeddings(self):
        # self.word_embeddings.requires_grad = True 
        self.word_embeddings.load_state_dict(self.emb_cp)


class MyDistilBertModel(DistilBertModel):
    
    def __init__(self, config):
        super(MyDistilBertModel, self).__init__(config)
        
        self.embeddings = MyEmbeddings(config)  # Embeddings
        self.transformer = MyTransformer(config)  # Encoder

        self.init_weights()
        self.config = config


    def set_custom_config(self, custom_config):
        self.custom_config = custom_config
        self.transformer.set_custom_config(custom_config)
        self.embeddings.set_custom_config(custom_config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        self.custom_config['devices'] = (torch.bernoulli(self.custom_config['active_probs']) > 0.5)
        self.custom_config['it'] = 0
        token_missed = torch.ones(self.config.vocab_size)
        vocab_dist = copy.deepcopy(self.custom_config['vocab_distribution'])

        # print(vocab_dist)
        # print(self.custom_config['devices'])

        vocab_dist[:, self.custom_config['devices'] == False] = 0
        vocab_dist = vocab_dist.sum(axis=1) > 0.5
        vocab_ids = self.custom_config['vocab_ids']
        # print(vocab_dist.sum())
        disable = [vocab_ids[t] for t in range(len(vocab_dist)) if vocab_dist[t] == False]
        # print(disable)
        self.embeddings.deactivate_embeddings(disable)
        self.set_custom_config(self.custom_config) # to broadcast the new config (including decive failure probs)
        # 1/0

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)

        self.embeddings.activate_all_embeddings()

        return self.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class MyDistilBert(DistilBertForSequenceClassification):

    def __init__(self, config):
        super(MyDistilBert, self).__init__(config)
        print(self.num_labels, self.classifier, self.pre_classifier, self.dropout)
        self.num_labels = config.num_labels
        self.distilbert = MyDistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.init_weights()
        
        


    def set_custom_config(self, custom_config):
        self.custom_config = custom_config
        self.distilbert.set_custom_config(custom_config)



def get_stoch_distilbert_for_classification(base="distilbert-base-uncased", num_labels = 5, output_attentions = False, output_hidden_states = False, custom_config = {'miss_ratio': 0}):
    model = MyDistilBert.from_pretrained(base, num_labels = num_labels, output_attentions = output_attentions, output_hidden_states = output_hidden_states)
    model.set_custom_config(custom_config)
    print(model.custom_config)
    return model

#-----------------------------------------------------
#-----------------------------------------------------
#-----------------------------------------------------


class MyBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super(MyBertSelfAttention, self).__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        #-------------------------
        miss_ids = torch.ones(self.num_attention_heads)<-1

        devices = self.custom_config['devices']
        ind = self.custom_config['distribution_matrix'][self.custom_config['it']]
        # print(ind)

        # print(miss_ids)
        # print('------------')
        for i in range(len(devices)):
            if devices[i] == False:
                miss_ids[ind==i] = True
        # print(miss_ids)
        
        # print(miss_ids)

        ## Zero Replacement
        context_layer[:,miss_ids,:,:] = 0
        
        #------------------------
        #------------------------

        # print(context_layer.shape)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # print(outputs[0].shape)
        # 1/0
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def set_custom_config(self, custom_config):
        self.custom_config = custom_config
        


class MyBertAttention(BertAttention):
    def __init__(self, config):
        super(MyBertAttention, self).__init__(config)
        self.self = MyBertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()
    
    def set_custom_config(self, custom_config):
        self.custom_config = custom_config
        self.self.set_custom_config(custom_config)
        

class MyBertLayer(BertLayer):
    def __init__(self, config):
        super(MyBertLayer, self).__init__(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MyBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = MyBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def set_custom_config(self, custom_config):
        self.custom_config = custom_config
        self.attention.set_custom_config(custom_config)
        if self.add_cross_attention:
            self.crossattention.set_custom_config(custom_config)




class MyBertEncoder(BertEncoder):
    def __init__(self, config):
        super(MyBertEncoder, self).__init__(config)
        self.config = config
        self.layer = nn.ModuleList([MyBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        self.custom_config['devices'] = (torch.bernoulli(self.custom_config['active_probs']) > 0.5)
        self.custom_config['it'] = 0
        # print(self.custom_config)
        self.set_custom_config(self.custom_config) # to broadcast the new config (including decive failure probs)
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            
            self.custom_config['it'] += 1
            self.set_custom_config(self.custom_config) # to broadcast the new config (including the new iterator)
        
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


    def set_custom_config(self, custom_config):
        self.custom_config = custom_config
        for l in self.layer:
            l.set_custom_config(custom_config)



class MyBertModel(BertModel):

    def __init__(self, config, add_pooling_layer=True):
        super(MyBertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = MyBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()
    
    def set_custom_config(self, custom_config):
        self.custom_config = custom_config
        self.encoder.set_custom_config(custom_config)



class MyBert(BertForSequenceClassification):

    def __init__(self, config):
        super(MyBert, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = MyBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def set_custom_config(self, custom_config):
        self.custom_config = custom_config
        self.bert.set_custom_config(custom_config)


def get_stoch_bert_for_classification(base="bert-base-uncased", num_labels = 5, output_attentions = False, output_hidden_states = False, custom_config = {'miss_ratio': 0}):
    model = MyBert.from_pretrained(base, num_labels = num_labels, output_attentions = output_attentions, output_hidden_states = output_hidden_states)
    model.set_custom_config(custom_config)
    print(model.custom_config)
    return model