import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# تنظیمات فونت
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'legend.fontsize': 12,
    'figure.dpi': 300
})

def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, **closed):
            return super().fill(closed=True, *args, **closed)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            return super()._gen_axes_spines()

    register_projection(RadarAxes)
    return theta

def parse_metric(content, metric_name, method):
    # الگوی جستجو برای پیدا کردن مقدار متریک خاص برای متد خاص
    # مثال: Stability Score: ... DRL : 0.995
    try:
        section = re.search(rf'{metric_name}:.*?{method}\s*:\s*([\d\.]+)', content, re.DOTALL)
        if section:
            return float(section.group(1))
    except:
        pass
    return 0.0

def get_data_for_method(filename, method_label):
    with open(filename, 'r') as f:
        content = f.read()
    
    # استخراج 5 متریک اصلی
    # توجه: برای Stability و Interruptions هرچه کمتر باشد بهتر است.
    # ما در نمودار رادار همه را نرمالایز می‌کنیم که "بیرون‌تر" بهتر باشد.
    
    stab = parse_metric(content, 'Stability Score', method_label)
    snr = parse_metric(content, 'Average SNR', method_label)
    cov = parse_metric(content, 'Coverage Ratio', method_label)
    inte = parse_metric(content, 'Service Interruptions', method_label)
    fair = parse_metric(content, 'Fairness Index', method_label)
    
    return [stab, snr, cov, inte, fair]

def main():
    # 1. خواندن داده‌ها
    # داده‌های خام (Raw Values)
    # ترتیب: Stability, SNR, Coverage, Interruptions, Fairness
    metrics = ['Stability\n(Inverted)', 'Average\nSNR', 'Coverage\nRatio', 'Service\nContinuity', 'Fairness\nIndex']
    
    # Proposed (DRL from proposed file)
    d_prop = get_data_for_method('results_proposed_final.txt', 'DRL')
    
    # MAB (MAB from proposed file)
    d_mab = get_data_for_method('results_proposed_final.txt', 'MAB')
    
    # Vanilla (DRL from vanilla file)
    d_van = get_data_for_method('results_vanilla_final.txt', 'DRL')
    
    # Ablation (DRL from ablation file)
    d_abl = get_data_for_method('results_ablation_final.txt', 'DRL')

    # Heuristic (Greedy from proposed file)
    d_gre = get_data_for_method('results_proposed_final.txt', 'GREEDY')

    data_raw = [d_prop, d_mab, d_van, d_abl, d_gre]
    
    # 2. نرمال‌سازی (Normalization) برای اینکه همه در رنج 0 تا 1 باشند و "بیشتر=بهتر" شود
    # برای Stability و Interruptions: معکوس می‌کنیم (1 / x) یا (Max - x)
    # اینجا از روش Min-Max هوشمند استفاده می‌کنیم
    
    data_norm = []
    for i in range(5): # Iterate over metrics
        col = [row[i] for row in data_raw]
        min_val = min(col)
        max_val = max(col)
        
        norm_col = []
        for val in col:
            if i == 0 or i == 3: # Stability & Interruptions (Lower is better)
                # فرمول معکوس: (Max - Val) / (Max - Min)
                # یعنی کمترین مقدار نمره 1 می‌گیرد (بهترین)
                if max_val == min_val: score = 1.0
                else: score = (max_val - val) / (max_val - min_val)
            else: # Higher is better
                if max_val == min_val: score = 1.0
                else: score = (val - min_val) / (max_val - min_val)
            
            # کمی هموارسازی برای جلوگیری از صفر مطلق
            score = 0.1 + (score * 0.9)
            norm_col.append(score)
        
        # ذخیره ستونی برای تبدیل به سطری
        if i == 0: normalized_data = [[x] for x in norm_col]
        else:
            for idx, val in enumerate(norm_col):
                normalized_data[idx].append(val)

    # 3. رسم نمودار
    N = 5
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    colors = ['#1f77b4', '#7f7f7f', '#ff7f0e', '#d62728', '#2ca02c']
    labels = ['Proposed (Balanced)', 'MAB Baseline', 'Vanilla DRL', 'Ablation (No Mem)', 'Greedy']
    styles = ['-', '--', ':', '-.', ':']
    markers = ['o', 's', '^', 'x', 'd']

    for d, color, label, style, marker in zip(normalized_data, colors, labels, styles, markers):
        ax.plot(theta, d, color=color, label=label, linewidth=2, linestyle=style, marker=marker, markersize=6)
        if label == 'Proposed (Balanced)':
            ax.fill(theta, d, facecolor=color, alpha=0.15) # فقط مال خودمان را پر رنگ کن

    ax.set_varlabels(metrics)
    
    # تنظیمات محورها
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=[], angle=0)
    
    # راهنما
    legend = ax.legend(loc=(0.9, .95), labelspacing=0.1, fontsize=12)
    
    plt.title('Holistic Performance Comparison\n(Outer is Better)', y=1.08, fontsize=16, fontweight='bold')
    
    plt.savefig('radar_chart_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Radar Chart generated: radar_chart_comparison.png")

if __name__ == "__main__":
    main()