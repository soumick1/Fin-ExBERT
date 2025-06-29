import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.nn import MultiheadAttention, GELU

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


class SentenceExtractionModel(nn.Module):
    def __init__(self,
                 base_model_name: str,
                 dropout_prob: float = 0.1,
                 adapter_dir: str = "./lora_finance_adapter",
                 backbone: str = 'default',
                 init_pos_frac: float = None    # NEW!
    ):
        """
        backbone:
          - 'default' → plain AutoModel.from_pretrained(base_model_name)
          - 'finexbert' → use the .bert submodule of your GraphAugmentedFinNLIModel
        """
        super().__init__()

        # load config
        config = AutoConfig.from_pretrained(base_model_name)

        if backbone == 'default':
            # plain BERT
            self.bert = AutoModel.from_pretrained(base_model_name, config=config)

        elif backbone == 'finexbert':
            # instantiate your full FinNLI model, then grab its .bert
            base_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_CLS"#"CAUSAL_LM",  # must match your fine-tune setting
            )
            full = get_peft_model(base_model, lora_cfg).to(DEVICE)
            chkpt_path = os.path.join(adapter_dir, "training_checkpoint.pt")
            if not os.path.isfile(chkpt_path):
                raise FileNotFoundError(f"No LoRA checkpoint at {chkpt_path}")
            ckpt = torch.load(chkpt_path, map_location=DEVICE)
            # ckpt["model_state_dict"] contains both base + LoRA weights; strict=False
            full.load_state_dict(ckpt["model_state_dict"], strict=False)
            # if you have a saved finexbert checkpoint, load it here:
            # full.load_state_dict(torch.load("path/to/finexbert.pth", map_location='cpu'))
            self.bert = full.base_model

        else:
            raise ValueError(f"Unknown backbone {backbone}")

        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

        # initialize bias to log-odds of init_pos_frac
        if init_pos_frac is not None:
            b0 = float(math.log(init_pos_frac / (1.0 - init_pos_frac)))
            self.classifier.bias.data.fill_(b0)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)
        x      = self.dropout(outputs.pooler_output)
        logits = self.classifier(x).squeeze(-1)   # [batch]
        return logits