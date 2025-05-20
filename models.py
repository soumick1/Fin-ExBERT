import torch
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
    def __init__(self, base_model_name="bert-base-uncased", hidden_dim=768, gnn_dim=128, freeze=True, lora_enabled=False):
        super().__init__()
        self.gnn_bert = GraphAugmentedNLIModel(
            base_model_name=base_model_name
        )
        if freeze:
            self.gnn_bert.load_state_dict(torch.load('gnn_model_weights_2.pt', weights_only=True))
            
        if lora_enabled:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_CLS",
                target_modules=['query', 'value']
            )
            
            # Wrap the model with LoRA again
            self.gnn_bert = get_peft_model(self.gnn_bert, lora_config)
            # Load the previously saved LoRA weights
            self.gnn_bert.load_adapter("./lora_finance_adapter", adapter_name="default")
            
            # Set LoRA adapter as active if needed (for PeftModel, it's usually active by default)
            self.gnn_bert.set_adapter("default")
        # Freeze all params in GNN-BERT
        if freeze:
            for param in self.gnn_bert.parameters():
                param.requires_grad = False

        # New trainable head
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
        hypothesis_node_indices=None
    ):
        with torch.no_grad():
            # We'll just get the last hidden states from BERT
            outputs = self.gnn_bert.bert(input_ids=input_ids, attention_mask=attention_mask)
            # shape: [batch_size, seq_len, hidden_dim]
            hidden_states = outputs.last_hidden_state

        # Now feed to the trainable span head
        start_logits, end_logits, no_span_logit = self.span_head(hidden_states, attention_mask)
        return {
            "start_logits": start_logits, 
            "end_logits": end_logits,
            "no_span_logit": no_span_logit
        }

class SparseGNN(nn.Module):
    """
    Simple multi-layer GNN using sparse adjacency for efficiency.
    Expects `edge_index` of shape [2, E] and node features `h` of shape [N, D].
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, activation=nn.ReLU):
        super().__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        # first layer
        self.linears.append(nn.Linear(in_dim, hidden_dim))
        # hidden layers
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        # output layer
        self.linears.append(nn.Linear(hidden_dim, out_dim))
        self.act = activation()

    def forward(self, h, edge_index):
        # h: [N, D]
        # edge_index: [2, E]
        N = h.size(0)
        E = edge_index.size(1)
        device = h.device
        # Create sparse adjacency with self-loops
        # Add self-loop edges
        self_loop = torch.arange(0, N, dtype=edge_index.dtype, device=device)
        self_loop = self_loop.unsqueeze(0).repeat(2, 1)
        full_index = torch.cat([edge_index, self_loop], dim=1)
        values = torch.ones(full_index.size(1), device=device)
        adj = torch.sparse_coo_tensor(full_index, values, (N, N))
        # message passing through layers
        x = h
        for lin in self.linears:
            # aggregate: sparse matrix multiplication
            x = torch.sparse.mm(adj, x)
            x = lin(x)
            x = self.act(x)
        return x

class FinExBERT(nn.Module):
    """
    Fin-ExBERT: BERT encoder + SparseGNN + span-prediction head
    """
    def __init__(self, bert_model, gnn_in, gnn_hidden, gnn_out, head_dim, num_gnn_layers=2):
        super().__init__()
        self.bert = bert_model
        # freeze BERT during GNN forward if desired
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # GNN module
        self.gnn = SparseGNN(gnn_in, gnn_hidden, gnn_out, num_layers=num_gnn_layers)
        # span-prediction head
        self.start_head = nn.Linear(gnn_out, head_dim)
        self.end_head = nn.Linear(gnn_out, head_dim)

    def forward(self, input_ids, attention_mask, edge_index):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]
        batch_size, seq_len, hidden_size = hidden_states.size()
        # reshape for GNN: flatten batch and tokens
        x = hidden_states.view(batch_size * seq_len, hidden_size)

        # Use same edge_index for each sequence: shift indices per batch
        all_x = []
        for b in range(batch_size):
            # offset edge indices by b * seq_len
            idx_shift = b * seq_len
            ei = edge_index + idx_shift
            x_b = x[b * seq_len:(b + 1) * seq_len]
            gnn_out = self.gnn(x_b, ei)
            all_x.append(gnn_out)
        gnn_feat = torch.stack(all_x, dim=0)  # [B, T, gnn_out]

        # span-prediction logits
        start_logits = self.start_head(gnn_feat)  # [B, T, head_dim]
        end_logits   = self.end_head(gnn_feat)
        return start_logits, end_logits