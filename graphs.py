import matplotlib.pyplot as plt
import numpy as np

# Ablation results
epochs = np.arange(1, 6)
loss_baseline = [1.1084, 1.1015, 1.1008, 0.9821, 1.0015]
loss_gnn      = [1.1044, 1.0503, 0.9025, 0.8319, 0.7638]

models = ['Baseline-BERT', 'GNN-Augmented-BERT']
accs   = [0.3232, 0.6937]
f1s    = [0.1628, 0.4184]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# (I) Loss vs. Epoch
ax1.plot(epochs, loss_baseline, marker='o', label='Baseline-BERT')
ax1.plot(epochs, loss_gnn,      marker='s', label='GNN-Augmented-BERT')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('Training Loss vs. Epoch \n(on 10% data and 5 epochs)', fontsize=14)
#ax1.ylim()
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=11)

# (II) Grouped Bar Chart for Accuracy and F1
x = np.arange(len(models))
width = 0.35

bars_acc = ax2.bar(x - width/2, accs, width, label='Accuracy')
bars_f1  = ax2.bar(x + width/2, f1s, width, label='F1', hatch='//')

# Annotations
for bar in bars_acc + bars_f1:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f'{height:.2f}',
        ha='center',
        va='bottom',
        fontsize=10
    )

ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_ylim(0, 1.0)
ax2.set_title('Validation Metrics Comparison \n(on 10% data and 5 epochs)', fontsize=14)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend(fontsize=11)

plt.show()
