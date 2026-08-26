"""
Visualization Package for Uncertainty Benchmark
可视化工具包

Available modules / 可用模块:
- reliability: Reliability diagrams / 可靠性图
- uncertainty: Uncertainty distribution plots / 不确定性分布图
- roc: ROC curve comparison / ROC曲线对比
- comparison: Multi-method comparison charts / 多方法对比图
- risk_coverage: Selective-classification curves / 风险-覆盖曲线
- noise_robustness: Separate noise severity curves / 噪声鲁棒性曲线
- seed_stability: Per-seed metric dispersion / 随机种子稳定性
"""
from .style import apply_publication_style

apply_publication_style()


from . import reliability
from . import uncertainty
from . import roc
from . import comparison
