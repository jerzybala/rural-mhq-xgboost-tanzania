"""
Create a summary infographic of the key findings
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure
fig = plt.figure(figsize=(16, 20))
fig.suptitle('XGBoost + SHAP Analysis: Lifestyle Drivers of Mental Health (MHQ) in Rural Tanzania', 
             fontsize=20, fontweight='bold', y=0.98)

# Color scheme
color_success = '#2ecc71'  # green
color_struggle = '#e74c3c'  # red
color_neutral = '#95a5a6'  # gray
color_highlight = '#3498db'  # blue

# === SECTION 1: Study Overview ===
ax1 = plt.subplot(6, 2, (1, 2))
ax1.axis('off')
ax1.text(0.5, 0.9, '📊 STUDY OVERVIEW', ha='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=color_highlight, alpha=0.3))

overview_text = """
Dataset: Rural Tanzania Global Mind Data
Sample Size: 5,095 participants
Classification: Binary (Struggling vs Succeeding)

• Struggling (MHQ < 0): 1,386 participants (27.2%)
• Succeeding (MHQ ≥ 100): 3,709 participants (72.8%)

Model: XGBoost Classifier + SHAP Analysis
ROC-AUC Score: 0.746 | Accuracy: 70%
"""
ax1.text(0.5, 0.4, overview_text, ha='center', va='center', fontsize=12,
         family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# === SECTION 2: Top Factors (Feature Importance) ===
ax2 = plt.subplot(6, 2, (3, 4))
ax2.set_title('🎯 TOP LIFESTYLE FACTORS PREDICTING MENTAL HEALTH', 
              fontsize=14, fontweight='bold', pad=20)

features = ['RelationWith\nAdultFamily', 'UPF.Freq', 'Smartphone\n.ownership', 
            'Education\nYears', 'Exercise\n.Freq', 'AgeOfFirst\nSP', 'ShareHome\nWith']
importance = [0.2906, 0.1589, 0.1317, 0.1166, 0.1113, 0.0993, 0.0916]
colors_bars = [color_success if i < 3 else color_neutral for i in range(len(features))]

y_pos = np.arange(len(features))
bars = ax2.barh(y_pos, importance, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(features, fontsize=11)
ax2.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 0.35)
ax2.invert_yaxis()

# Add percentage labels
for i, (bar, imp) in enumerate(zip(bars, importance)):
    ax2.text(imp + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{imp*100:.1f}%', va='center', fontsize=11, fontweight='bold')

ax2.grid(axis='x', alpha=0.3)
ax2.text(0.5, -0.15, '⭐ Top 3 factors account for 57% of predictive power', 
         ha='center', transform=ax2.transAxes, fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# === SECTION 3: Key Finding #1 - Family Relationships ===
ax3 = plt.subplot(6, 2, 5)
ax3.axis('off')
ax3.add_patch(FancyBboxPatch((0.05, 0.1), 0.9, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color_success, alpha=0.2, edgecolor=color_success, linewidth=3))
ax3.text(0.5, 0.85, '🏆 #1 PREDICTOR', ha='center', fontsize=14, fontweight='bold', color=color_success)
ax3.text(0.5, 0.7, 'FAMILY RELATIONSHIPS', ha='center', fontsize=13, fontweight='bold')
ax3.text(0.5, 0.5, '29.1% Importance', ha='center', fontsize=20, fontweight='bold', color=color_success)
ax3.text(0.5, 0.25, '"Very Close to Most"\nfamily members', ha='center', fontsize=11, style='italic')
ax3.text(0.5, 0.05, '→ Strongest predictor of\nSucceeding MHQ', ha='center', fontsize=10)

# === SECTION 4: Key Finding #2 - UPF Consumption ===
ax4 = plt.subplot(6, 2, 6)
ax4.axis('off')
ax4.add_patch(FancyBboxPatch((0.05, 0.1), 0.9, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color_struggle, alpha=0.2, edgecolor=color_struggle, linewidth=3))
ax4.text(0.5, 0.85, '🥗 #2 PREDICTOR', ha='center', fontsize=14, fontweight='bold', color=color_struggle)
ax4.text(0.5, 0.7, 'UPF CONSUMPTION', ha='center', fontsize=13, fontweight='bold')
ax4.text(0.5, 0.5, '15.9% Importance', ha='center', fontsize=20, fontweight='bold', color=color_struggle)
ax4.text(0.5, 0.25, '"Rarely/Never" consuming\nultra-processed foods', ha='center', fontsize=11, style='italic')
ax4.text(0.5, 0.05, '→ Associated with better\nmental health', ha='center', fontsize=10)

# === SECTION 5: SHAP Impact Rankings ===
ax5 = plt.subplot(6, 2, (7, 8))
ax5.set_title('🔍 SHAP VALUES: Actual Predictive Impact', fontsize=14, fontweight='bold', pad=20)

shap_features = ['RelationWith\nAdultFamily', 'UPF.Freq', 'Education\nYears', 
                'Exercise.Freq', 'AgeOfFirst\nSP', 'ShareHome\nWith', 'Smartphone\n.ownership']
shap_values = [0.4805, 0.2706, 0.1999, 0.1900, 0.1555, 0.1495, 0.0928]
colors_shap = [color_success if i < 2 else color_highlight if i < 4 else color_neutral 
               for i in range(len(shap_features))]

y_pos_shap = np.arange(len(shap_features))
bars_shap = ax5.barh(y_pos_shap, shap_values, color=colors_shap, alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.set_yticks(y_pos_shap)
ax5.set_yticklabels(shap_features, fontsize=11)
ax5.set_xlabel('Mean Absolute SHAP Value (Impact Magnitude)', fontsize=12, fontweight='bold')
ax5.set_xlim(0, 0.55)
ax5.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars_shap, shap_values)):
    ax5.text(val + 0.015, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

ax5.grid(axis='x', alpha=0.3)
ax5.text(0.5, -0.12, 'SHAP reveals the actual magnitude of each feature\'s impact on predictions', 
         ha='center', transform=ax5.transAxes, fontsize=10, style='italic')

# === SECTION 6: Model Performance ===
ax6 = plt.subplot(6, 2, 9)
ax6.axis('off')
ax6.text(0.5, 0.95, '📈 MODEL PERFORMANCE', ha='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=color_highlight, alpha=0.3))

perf_text = """
ROC-AUC Score: 0.746
Cross-Val ROC-AUC: 0.710 ± 0.011

Overall Accuracy: 70%

Precision (Succeeding): 84%
Recall (Succeeding): 73%

Precision (Struggling): 47%
Recall (Struggling): 64%
"""
ax6.text(0.5, 0.45, perf_text, ha='center', va='center', fontsize=11,
         family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# === SECTION 7: Confusion Matrix Simplified ===
ax7 = plt.subplot(6, 2, 10)
ax7.axis('off')
ax7.text(0.5, 0.95, '🎯 PREDICTIONS', ha='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor=color_highlight, alpha=0.3))

# Simple visual representation
ax7.text(0.25, 0.65, 'Struggling\n(actual)', ha='center', fontsize=10, fontweight='bold')
ax7.text(0.25, 0.5, '✓ 177', ha='center', fontsize=14, color=color_success, fontweight='bold')
ax7.text(0.25, 0.4, '✗ 100', ha='center', fontsize=12, color=color_struggle)

ax7.text(0.75, 0.65, 'Succeeding\n(actual)', ha='center', fontsize=10, fontweight='bold')
ax7.text(0.75, 0.5, '✓ 541', ha='center', fontsize=14, color=color_success, fontweight='bold')
ax7.text(0.75, 0.4, '✗ 201', ha='center', fontsize=12, color=color_struggle)

ax7.text(0.5, 0.15, 'Test Set: 1,019 participants', ha='center', fontsize=10, style='italic')

# === SECTION 8: Key Insights ===
ax8 = plt.subplot(6, 2, (11, 12))
ax8.axis('off')
ax8.text(0.5, 0.95, '💡 KEY INSIGHTS & RECOMMENDATIONS', ha='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

insights = """
1. FAMILY RELATIONSHIPS ARE PARAMOUNT
   • Being "Very Close to Most" family members is the #1 protective factor
   • Social disconnection strongly predicts mental health struggles
   → Recommendation: Prioritize family therapy and community support programs

2. NUTRITION MATTERS FOR MENTAL HEALTH
   • Lower ultra-processed food consumption correlates with better MHQ
   • The relationship appears dose-dependent
   → Recommendation: Nutrition education and access to whole foods

3. EDUCATION IS PROTECTIVE
   • More years of education associated with better mental health outcomes
   • Likely through multiple pathways (coping, economic, social, health literacy)
   → Recommendation: Expand educational opportunities in rural areas

4. LIFESTYLE FACTORS ARE MODIFIABLE
   • Exercise frequency shows moderate positive impact
   • Multiple factors can be addressed through interventions
   → Recommendation: Integrated approach targeting multiple factors

5. NEED FOR HOLISTIC APPROACH
   • Top factors span social, nutritional, educational, and physical domains
   • Single-factor interventions may be insufficient
   → Recommendation: Comprehensive rural mental health programs
"""

ax8.text(0.05, 0.80, insights, ha='left', va='top', fontsize=10.5,
         family='sans-serif', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('SUMMARY_INFOGRAPHIC.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✅ Summary infographic created: SUMMARY_INFOGRAPHIC.png")
print("   This provides a visual overview of all key findings!")
