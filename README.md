<div class="Box-sc-g0xbh4-0 bJMeLZ js-snippet-clipboard-copy-unpositioned" data-hpc="true"><article class="markdown-body entry-content container-lg" itemprop="text"><div class="markdown-heading" dir="auto"><h1 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">JAX-CFD：JAX 中的计算流体动力学</font></font></h1><a id="user-content-jax-cfd-computational-fluid-dynamics-in-jax" class="anchor" aria-label="永久链接：JAX-CFD：JAX 中的计算流体动力学" href="#jax-cfd-computational-fluid-dynamics-in-jax"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">作者：德米特里·科奇科夫、杰米·A·史密斯、彼得·诺加德、吉迪恩·德累斯顿、阿亚·阿利瓦、斯蒂芬·霍耶</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">JAX-CFD 是一个实验研究项目，旨在探索机器学习、自动微分和硬件加速器 (GPU/TPU) 在计算流体动力学方面的潜力。它是在
</font></font><a href="https://github.com/google/jax"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">JAX</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">中实现的。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">要了解有关我们一般方法的更多信息，请阅读我们的论文</font></font><a href="https://www.pnas.org/content/118/21/e2101784118" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">机器学习加速计算流体动力学</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">
(PNAS 2021)。</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">入门</font></font></h2><a id="user-content-getting-started" class="anchor" aria-label="永久链接：开始使用" href="#getting-started"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">“notebooks”目录包含几个使用 JAX-CFD 代码的演示。</font></font></p>
<ul dir="auto">
<li>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">不同模拟设置的演示：</font></font></p>
<ul dir="auto">
<li><a href="https://colab.research.google.com/github/google/jax-cfd/blob/main/notebooks/demo.ipynb" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在交错网格上使用 FVM 进行 2D 仿真</font></font></a></li>
<li><a href="https://colab.research.google.com/github/google/jax-cfd/blob/main/notebooks/spectral_forced_turbulence.ipynb" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">使用伪谱解算器进行 2D 模拟</font></font></a></li>
<li><a href="https://colab.research.google.com/github/google/jax-cfd/blob/main/notebooks/channel_flow_demo.ipynb" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">渠道流动的二维模拟</font></font></a></li>
<li><a href="https://colab.research.google.com/github/google/jax-cfd/blob/main/notebooks/collocated_demo.ipynb" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">在并置网格上使用 FVM 进行 2D 仿真</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">（实验）</font></font></li>
</ul>
</li>
<li>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">重现我们的 PNAS 论文的结果：</font></font></p>
<ul dir="auto">
<li><a href="https://colab.research.google.com/github/google/jax-cfd/blob/main/notebooks/ml_accelerated_cfd_data_analysis.ipynb" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">数据分析与评估</font></font></a></li>
<li><a href="https://colab.research.google.com/github/google/jax-cfd/blob/main/notebooks/ml_model_inference_demo.ipynb" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">运行我们预先训练的模型</font></font></a></li>
</ul>
</li>
</ul>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">组织</font></font></h2><a id="user-content-organization" class="anchor" aria-label="永久链接： 组织" href="#organization"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">JAX-CFD 围绕子模块进行组织：</font></font></p>
<ul dir="auto">
<li><code>jax_cfd.base</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：CFD 的核心有限体积/差分方法，用 JAX 编写。</font></font></li>
<li><code>jax_cfd.spectral</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：CFD 的核心伪谱方法，用 JAX 编写。</font></font></li>
<li><code>jax_cfd.ml</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：CFD 机器学习增强模型，用 JAX 和</font></font><a href="https://dm-haiku.readthedocs.io/en/latest/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Haiku</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">编写。</font></font></li>
<li><code>jax_cfd.data</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：数据处理实用程序，用于准备、评估和后处理使用 JAX-CFD 创建的数据，用
</font></font><a href="http://xarray.pydata.org/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Xarray</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">和
</font></font><a href="https://pillow.readthedocs.io/" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Pillow</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">编写。</font></font></li>
</ul>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">基本安装</font></font><code>pip install jax-cfd</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">仅需要 NumPy、SciPy 和 JAX。要安装其他子模块的依赖项，请使用</font></font><code>pip install jax-cfd[ml]</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">,
</font></font><code>pip install jax-cfd[data]</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">或</font></font><code>pip install jax-cfd[complete]</code><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">数值</font></font></h2><a id="user-content-numerics" class="anchor" aria-label="固定链接： 数值" href="#numerics"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">JAX-CFD 目前专注于非定常湍流：</font></font></p>
<ul dir="auto">
<li><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">空间离散化</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：
</font></font><ul dir="auto">
<li><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">交错网格（“Arakawa C”或“MAC”网格）上的有限体积/差分</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">方法，每个单元中心的压力和相应面上定义的速度分量。</font></font></li>
<li><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">涡度的伪谱</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">方法使用非线性项的抗混叠滤波技术来保持稳定性。</font></font></li>
</ul>
</li>
<li><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">时间离散化</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：目前只有一阶时间离散化，使用显式时间步进进行平流，使用隐式或显式时间步进进行扩散。</font></font></li>
<li><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">压力求解</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：CG 或实值 FFT 的快速对角化（适用于周期性边界条件）。</font></font></li>
<li><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">边界条件</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：目前仅支持周期性边界条件。</font></font></li>
<li><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">平流</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：我们实施二阶精确的“Van Leer”方案。</font></font></li>
<li><em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">闭包</font></font></em><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">：我们目前实施 Smagorinsky 涡流粘度模型。</font></font></li>
</ul>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">TODO：添加一个笔记本，更深入地解释我们的数值模型。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">从长远来看，我们有兴趣扩展 JAX-CFD 以实现与相关研究相关的方法，例如，</font></font></p>
<ul dir="auto">
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">共置网格</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">替代边界条件（例如，非周期边界和浸入边界方法）</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">高阶时间步长</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">几何多重网格</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">稳态模拟（例如 RANS）</font></font></li>
<li><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">跨多个 TPU/GPU 的分布式模拟</font></font></li>
</ul>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我们欢迎在这些方面进行合作！在开始重要工作之前，请联系（通过 GitHub 或通过电子邮件）进行协调。</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">使用 JAX-CFD 的项目</font></font></h2><a id="user-content-projects-using-jax-cfd" class="anchor" aria-label="永久链接：使用 JAX-CFD 的项目" href="#projects-using-jax-cfd"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<ul dir="auto">
<li><a href="https://github.com/googleinterns/invobs-data-assimilation"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">使用学习逆观测算子进行变分数据同化</font></font></a></li>
</ul>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">其他很棒的项目</font></font></h2><a id="user-content-other-awesome-projects" class="anchor" aria-label="永久链接：其他很棒的项目" href="#other-awesome-projects"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">其他与深度学习兼容的可微分 CFD 代码：</font></font></p>
<ul dir="auto">
<li><a href="https://github.com/tum-pbs/PhiFlow/"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">PhiFlow</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">支持 TensorFlow、PyTorch 和 JAX</font></font></li>
<li><a href="https://github.com/HIPS/autograd#end-to-end-examples"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Autograd 中的流体模拟</font></font></a></li>
</ul>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">JAX 科学：</font></font></p>
<ul dir="auto">
<li><a href="https://github.com/google/jax-md"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">JAX-MD</font></font></a></li>
<li><a href="https://github.com/google-research/google-research/tree/master/jax_dft"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">JAX-DFT</font></font></a></li>
<li><a href="https://github.com/DifferentiableUniverseInitiative/jax_cosmo"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">贾克斯·科斯莫</font></font></a></li>
<li><a href="https://github.com/team-ocean/veros"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">维罗斯</font></font></a></li>
</ul>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我们错过了什么吗？请告诉我们！</font></font></p>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">引文</font></font></h2><a id="user-content-citation" class="anchor" aria-label="永久链接：引文" href="#citation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">如果您使用我们的有限体积法 (FVM) 或 ML 模型，请引用：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>@article{Kochkov2021-ML-CFD,
  author = {Kochkov, Dmitrii and Smith, Jamie A. and Alieva, Ayya and Wang, Qing and Brenner, Michael P. and Hoyer, Stephan},
  title = {Machine learning{\textendash}accelerated computational fluid dynamics},
  volume = {118},
  number = {21},
  elocation-id = {e2101784118},
  year = {2021},
  doi = {10.1073/pnas.2101784118},
  publisher = {National Academy of Sciences},
  issn = {0027-8424},
  URL = {https://www.pnas.org/content/118/21/e2101784118},
  eprint = {https://www.pnas.org/content/118/21/e2101784118.full.pdf},
  journal = {Proceedings of the National Academy of Sciences}
}
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="@article{Kochkov2021-ML-CFD,
  author = {Kochkov, Dmitrii and Smith, Jamie A. and Alieva, Ayya and Wang, Qing and Brenner, Michael P. and Hoyer, Stephan},
  title = {Machine learning{\textendash}accelerated computational fluid dynamics},
  volume = {118},
  number = {21},
  elocation-id = {e2101784118},
  year = {2021},
  doi = {10.1073/pnas.2101784118},
  publisher = {National Academy of Sciences},
  issn = {0027-8424},
  URL = {https://www.pnas.org/content/118/21/e2101784118},
  eprint = {https://www.pnas.org/content/118/21/e2101784118.full.pdf},
  journal = {Proceedings of the National Academy of Sciences}
}" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">如果您使用我们的光谱代码，请引用：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>@article{Dresdner2022-Spectral-ML,
  doi = {10.48550/ARXIV.2207.00556},
  url = {https://arxiv.org/abs/2207.00556},
  author = {Dresdner, Gideon and Kochkov, Dmitrii and Norgaard, Peter and Zepeda-Núñez, Leonardo and Smith, Jamie A. and Brenner, Michael P. and Hoyer, Stephan},
  title = {Learning to correct spectral methods for simulating turbulent flows},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="@article{Dresdner2022-Spectral-ML,
  doi = {10.48550/ARXIV.2207.00556},
  url = {https://arxiv.org/abs/2207.00556},
  author = {Dresdner, Gideon and Kochkov, Dmitrii and Norgaard, Peter and Zepeda-Núñez, Leonardo and Smith, Jamie A. and Brenner, Michael P. and Hoyer, Stephan},
  title = {Learning to correct spectral methods for simulating turbulent flows},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">当地发展</font></font></h2><a id="user-content-local-development" class="anchor" aria-label="永久链接：地方发展" href="#local-development"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">本地安装以进行开发：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>git clone https://github.com/google/jax-cfd.git
cd jax-cfd
pip install jaxlib
pip install -e ".[complete]"
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="git clone https://github.com/google/jax-cfd.git
cd jax-cfd
pip install jaxlib
pip install -e &quot;.[complete]&quot;" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">然后手动运行测试套件：</font></font></p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>pytest -n auto jax_cfd --dist=loadfile --ignore=jax_cfd/base/validation_test.py
</code></pre><div class="zeroclipboard-container">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn btn-invisible js-clipboard-copy m-2 p-0 tooltipped-no-delay d-flex flex-justify-center flex-items-center" data-copy-feedback="Copied!" data-tooltip-direction="w" value="pytest -n auto jax_cfd --dist=loadfile --ignore=jax_cfd/base/validation_test.py" tabindex="0" role="button">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none">
    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
</svg>
    </clipboard-copy>
  </div></div>
</article></div>
