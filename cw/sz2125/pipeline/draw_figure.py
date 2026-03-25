"""
Due to the significant memory overhead and VRAM limitations when training and 
distilling large Transformer models (e.g., BERT), attempting to generate plots 
directly in the same Jupyter Notebook after the training loop often causes the 
environment to freeze or crash with Out-Of-Memory (OOM) errors. 

To ensure stability and successfully generate the visualizations, the final 
accuracy metrics from each experimental phase were carefully recorded and are 
plotted in this standalone script.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Organize the experimental data
data = {
    'Model / Configuration': [
        'Teacher Baseline\n(bert-base)',
        'Student Pre-pruning\n(bert-tiny)',
        'Student Post-pruning\n(50% Sparsity)',
        'Student Post-pruning\n+ FT',
        'Strategy A\n(Logits-only)',
        'Strategy B\n(Logits+Hidden)',
        'Strategy C\n(Logits+Hidden+Attn)',
        'Strategy D\n(Logits-only + FT)'
    ],
    'Accuracy': [
        0.90654,
        0.78573,
        0.65897,
        0.80254,
        0.83988,
        0.83768,
        0.83556,
        0.85745
    ]
}

df = pd.DataFrame(data)

# 2. Set the plotting style (white background with grid lines)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 7))

# 3. Define the color palette (cool grays for baselines, greens for KD, orange for the best)
colors = ['#B0BEC5', '#90A4AE', '#78909C', '#546E7A', '#81C784', '#66BB6A', '#43A047', '#FF9800']

# 4. Plot the vertical bar chart
ax = sns.barplot(x='Model / Configuration', y='Accuracy', data=df, palette=colors)

# 5. Set the title and axis labels
plt.title('BERT Top-1 Accuracy on IMDB Dataset Across Different Configurations', fontsize=16, pad=20, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('', fontsize=12)

# 6. Set the Y-axis display limits to highlight differences
plt.ylim(0.60, 0.95)

# 7. Rotate X-axis labels to prevent long text from overlapping
plt.xticks(rotation=30, ha='right', fontsize=11, fontweight='bold')

# 8. Add data labels on top of each bar
for p in ax.patches:
    height = p.get_height()
    plt.text(p.get_x() + p.get_width() / 2., height + 0.005,
             f'{height:.5f}',
             ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')

plt.tight_layout()

# 9. Save the figure as a high-resolution JPG image
plt.savefig('bert_kd_comparison_vertical.jpg', format='jpg', dpi=300, bbox_inches='tight')

# Display the plot (if running in a notebook/interactive window)
plt.show()