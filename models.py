import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESSED_DIR = "preprocessed_snli"
MIXED_PRECISION = "fp16"


class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, node_embeddings, edges):
        if node_embeddings.size(0) == 0:
            return torch.zeros(1, self.fc.out_features, device=node_embeddings.device)
        num_nodes = node_embeddings.size(0)
        adj = torch.zeros((num_nodes, num_nodes), device=node_embeddings.device)
        for (src, dst) in edges:
            if src < num_nodes and dst < num_nodes:
                adj[src, dst] = 1.0
        deg = adj.sum(dim=1, keepdim=True) + 1e-10
        adj_norm = adj / deg
        agg_embeddings = adj_norm @ node_embeddings
        return F.relu(self.fc(agg_embeddings))


class GraphAugmentedNLIModel(nn.Module):
    def __init__(self, base_model_name, num_labels=3, hidden_dim=768, gnn_dim=128):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model_name)
        config.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(base_model_name, config=config)
        self.dropout = nn.Dropout(0.1)

        self.gnn_premise = SimpleGNN(hidden_dim, gnn_dim)
        self.gnn_hypothesis = SimpleGNN(hidden_dim, gnn_dim)

        self.classifier = nn.Linear(hidden_dim + gnn_dim*2, num_labels)

    def forward(self, input_ids, attention_mask, premise_graph_tokens, premise_graph_edges, premise_node_indices,
                hypothesis_graph_tokens, hypothesis_graph_edges, hypothesis_node_indices, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:,0,:]  # [batch, hidden_dim]

        batch_size = input_ids.size(0)
        gnn_p_outputs = []
        gnn_h_outputs = []

        # Now node indices are precomputed. We just take those embeddings directly.
        # node_indices correspond to the positions in input_ids whose embeddings represent that node.
        for i in range(batch_size):
            instance_hidden = outputs.last_hidden_state[i]  # [seq_len, hidden_dim]

            p_edges = premise_graph_edges[i]
            p_indices = premise_node_indices[i]
            h_edges = hypothesis_graph_edges[i]
            h_indices = hypothesis_node_indices[i]

            # Gather node embeddings
            p_nodes = instance_hidden[p_indices] if len(p_indices) > 0 else torch.empty(0, instance_hidden.size(-1), device=instance_hidden.device)
            h_nodes = instance_hidden[h_indices] if len(h_indices) > 0 else torch.empty(0, instance_hidden.size(-1), device=instance_hidden.device)

            p_gnn_out = self.gnn_premise(p_nodes, p_edges) if p_nodes.size(0) > 0 else torch.zeros(1,128, device=DEVICE)
            h_gnn_out = self.gnn_hypothesis(h_nodes, h_edges) if h_nodes.size(0) > 0 else torch.zeros(1,128, device=DEVICE)

            p_mean = p_gnn_out.mean(dim=0, keepdim=True)
            h_mean = h_gnn_out.mean(dim=0, keepdim=True)

            gnn_p_outputs.append(p_mean)
            gnn_h_outputs.append(h_mean)

        gnn_p_outputs = torch.cat(gnn_p_outputs, dim=0) # [batch, gnn_dim]
        gnn_h_outputs = torch.cat(gnn_h_outputs, dim=0) # [batch, gnn_dim]

        fused = torch.cat([cls_embedding, gnn_p_outputs, gnn_h_outputs], dim=-1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}



class SimpleFinGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, node_embeddings, edges):
        if node_embeddings.size(0) == 0:
            return torch.zeros(1, self.fc.out_features, device=node_embeddings.device)
        num_nodes = node_embeddings.size(0)
        adj = torch.zeros((num_nodes, num_nodes), device=node_embeddings.device)
        for (src, dst) in edges:
            if src < num_nodes and dst < num_nodes:
                adj[src, dst] = 1.0
        deg = adj.sum(dim=1, keepdim=True) + 1e-10
        adj_norm = adj / deg
        agg_embeddings = adj_norm @ node_embeddings
        return F.relu(self.fc(agg_embeddings))


class GraphAugmentedFinNLIModel(nn.Module):
    def __init__(self, base_model_name, num_labels=3, hidden_dim=768, gnn_dim=128):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model_name)
        config.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(base_model_name, config=config)
        self.dropout = nn.Dropout(0.1)

        self.gnn_premise = SimpleGNN(hidden_dim, gnn_dim)
        self.gnn_hypothesis = SimpleGNN(hidden_dim, gnn_dim)

        self.classifier = nn.Linear(hidden_dim + gnn_dim*2, num_labels)
        self.config = self.bert.config
        self.config.num_labels = num_labels

    def forward(self,
            input_ids=None,
            attention_mask=None,
            premise_graph_tokens=None,
            hypothesis_graph_tokens=None,
            premise_graph_edges=None,
            hypothesis_graph_edges=None,
            premise_node_indices=None,
            hypothesis_node_indices=None,
            labels=None,
            inputs_embeds=None,
            **kwargs):
        # Even if we don't use inputs_embeds, we should pass it into self.bert call:
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            inputs_embeds=inputs_embeds,
                            **{k:v for k,v in kwargs.items() if k in self.bert.forward.__code__.co_varnames})
    
        cls_embedding = outputs.last_hidden_state[:,0,:]  # [batch, hidden_dim]
    
        batch_size = input_ids.size(0) if input_ids is not None else outputs.last_hidden_state.size(0)
        gnn_p_outputs = []
        gnn_h_outputs = []
    
        for i in range(batch_size):
            instance_hidden = outputs.last_hidden_state[i]  # [seq_len, hidden_dim]
    
            p_edges = premise_graph_edges[i]
            p_indices = premise_node_indices[i]
            h_edges = hypothesis_graph_edges[i]
            h_indices = hypothesis_node_indices[i]
    
            p_nodes = instance_hidden[p_indices] if len(p_indices) > 0 else torch.empty(0, instance_hidden.size(-1), device=instance_hidden.device)
            h_nodes = instance_hidden[h_indices] if len(h_indices) > 0 else torch.empty(0, instance_hidden.size(-1), device=instance_hidden.device)
    
            p_gnn_out = self.gnn_premise(p_nodes, p_edges) if p_nodes.size(0) > 0 else torch.zeros(1,128, device=instance_hidden.device)
            h_gnn_out = self.gnn_hypothesis(h_nodes, h_edges) if h_nodes.size(0) > 0 else torch.zeros(1,128, device=instance_hidden.device)
    
            p_mean = p_gnn_out.mean(dim=0, keepdim=True)
            h_mean = h_gnn_out.mean(dim=0, keepdim=True)
    
            gnn_p_outputs.append(p_mean)
            gnn_h_outputs.append(h_mean)
    
        gnn_p_outputs = torch.cat(gnn_p_outputs, dim=0) # [batch, gnn_dim]
        gnn_h_outputs = torch.cat(gnn_h_outputs, dim=0) # [batch, gnn_dim]
    
        fused = torch.cat([cls_embedding, gnn_p_outputs, gnn_h_outputs], dim=-1)
        logits = self.classifier(fused)
    
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

'''
class SpanExtractionHead(nn.Module):
    """
    Predicts start index, end index, and a 'no span' logit from
    the final hidden states of the encoder (e.g., GNN-BERT).
    """
    def __init__(self, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.start_classifier = nn.Linear(hidden_dim, 1)
        self.end_classifier   = nn.Linear(hidden_dim, 1)
        self.no_span_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.dropout(hidden_states)  # [batch_size, seq_len, hidden_dim]
        start_logits = self.start_classifier(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        end_logits   = self.end_classifier(hidden_states).squeeze(-1)    # [batch_size, seq_len]

        # For 'no span' detection, we use the [CLS] token embedding
        cls_hidden = hidden_states[:, 0, :]  # shape [batch_size, hidden_dim]
        no_span_logit = self.no_span_classifier(cls_hidden).squeeze(-1)  # [batch_size]

        # If attention_mask is provided, mask out invalid positions
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(attention_mask == 0, -1e4)
            end_logits   = end_logits.masked_fill(attention_mask == 0, -1e4)

        return start_logits, end_logits, no_span_logit
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
 
class SpanExtractionHead(nn.Module):
    """
    Deeper version that includes multiple linear+ReLU layers
    before producing start/end/no_span logits.
    """
    def __init__(self, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # A small feedforward "projection" stack,
        # which preserves the [batch, seq_len, hidden_dim] shape.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
        )
        
        # Final classifiers
        self.start_classifier = nn.Linear(hidden_dim, 1)      # -> [batch, seq_len, 1]
        self.end_classifier   = nn.Linear(hidden_dim, 1)      # -> [batch, seq_len, 1]
        self.no_span_classifier = nn.Linear(hidden_dim, 1)    # -> [batch, 1]
 
    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: optional [batch_size, seq_len]
        
        Returns:
          start_logits: [batch_size, seq_len]
          end_logits:   [batch_size, seq_len]
          no_span_logit: [batch_size]
        """
        # (1) Dropout on input
        x = self.dropout(hidden_states)
        
        # (2) Pass through the MLP, shape remains [batch, seq_len, hidden_dim]
        x = self.mlp(x)  
        
        # (3) Compute start/end logits for each token
        start_logits = self.start_classifier(x).squeeze(-1)  # [batch_size, seq_len]
        end_logits   = self.end_classifier(x).squeeze(-1)    # [batch_size, seq_len]
        
        # (4) Compute "no span" from the [CLS] hidden state (token index = 0)
        cls_hidden = x[:, 0, :]            # [batch_size, hidden_dim]
        no_span_logit = self.no_span_classifier(cls_hidden).squeeze(-1)  # [batch_size]
        
        # (5) Optional: mask out invalid tokens
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(attention_mask == 0, -1e4)
            end_logits   = end_logits.masked_fill(attention_mask == 0, -1e4)
        
        return start_logits, end_logits, no_span_logit

class FrozenGNNBertSpanModel(nn.Module):
    """
    Wraps the GNN-augmented BERT, freezes its parameters,
    and adds a span-extraction head on top.
    """
    def __init__(
        self,
        base_model_name: str = "bert-base-uncased",
        hidden_dim: int = 768,
        gnn_dim: int = 128,
        freeze: bool = True,
        lora_enabled: bool = False,
        gnn_ckpt_path: str = "gnn_model_checkpoint.pt",
    ):
        super().__init__()
        # (1) build the same GNN-BERT backbone
        self.gnn_bert = GraphAugmentedNLIModel(base_model_name=base_model_name)

        # (2) if requested, load & freeze its weights
        if freeze:
            ckpt = torch.load(gnn_ckpt_path, map_location=DEVICE)
            # pull out only the model_state_dict if present
            state = ckpt.get("model_state_dict", ckpt)
            # we allow mismatches (e.g. optimizer keys) by strict=False
            self.gnn_bert.load_state_dict(state, strict=False)
            # freeze all backbone params
            for p in self.gnn_bert.parameters():
                p.requires_grad = False

        # (3) if you also want a LoRA adapter on the backbone:
        if lora_enabled:
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_CLS",
                target_modules=["query", "value"],
            )
            self.gnn_bert = get_peft_model(self.gnn_bert, lora_cfg)
            # assume your adapter .pt is in lora_finance_adapter/training_checkpoint.pt
            adapter_ckpt = torch.load(
                os.path.join("lora_finance_adapter", "training_checkpoint.pt"),
                map_location=DEVICE,
            )
            self.gnn_bert.load_state_dict(adapter_ckpt.get("model_state_dict", adapter_ckpt), strict=False)
            self.gnn_bert.set_adapter("default")
            # freeze base again, leave LoRA tunables on
            for name, p in self.gnn_bert.named_parameters():
                if "lora_" not in name:
                    p.requires_grad = False

        # (4) finally our new span-prediction head
        self.span_head = SpanExtractionHead(hidden_dim=hidden_dim)

    def forward(
        self,
        input_ids,
        attention_mask,
        premise_graph_tokens=None,
        premise_graph_edges=None,
        premise_node_indices=None,
        hypothesis_graph_tokens=None,
        hypothesis_graph_edges=None,
        hypothesis_node_indices=None,
    ):
        # get frozen backbone outputs
        with torch.no_grad():
            bert_out = self.gnn_bert.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = bert_out.last_hidden_state  # [B, T, H]

        # span‐head on top
        start_logits, end_logits, no_span_logit = self.span_head(hidden_states, attention_mask)
        return {
            "start_logits": start_logits,
            "end_logits":   end_logits,
            "no_span_logit": no_span_logit,
        }
