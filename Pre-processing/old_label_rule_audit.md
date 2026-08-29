# 旧版 Primary_or_Metastatic / metastasis_label 生成规则逐数据集审计

本文档逐条核对了 `Pre-processing_BRCA/COAD/LUCA/OVC.ipynb` 里每个数据集的原始代码，
记录旧规则实际是怎么写的（不是猜测，是读代码得到的）。目的：

1. 给审稿人2要求的"per-dataset truth table"提供代码层面的依据；
2. 记录每个数据集里，"患者取样时是否已确诊远处转移（M1/FIGO IV）"这个信息是否可得，
   供后续决定要不要把这个维度重新纳入标签体系。

**重要提醒**：最终存进 atlas 的三分类字段 `metastasis_label`（No_Mets/Regional_Mets/
Distant_Mets）在仓库任何代码里都找不到生成它的逻辑——下表里"旧规则"指的是能追溯到的
`Primary_or_Metastatic`（二分类）赋值逻辑，这是唯一有代码可查的中间字段。`metastasis_label`
与 `Primary_or_Metastatic` 之间的换算关系无法从代码复现，只能通过和 `Final_tissue`/
`Final_patient_stage` 做交叉表反推（详见对话记录，已用多个数据集验证：换算时明显读取了
`Final_patient_stage`，而不是这里列的 `Primary_or_Metastatic`）。

---

## 乳腺癌 (Breast Cancer)

| Project_ID | 组织赋值 | Primary_or_Metastatic 旧规则（代码原文行为） | 取样时是否已知远处转移(M1/IV)可得？ | 备注 |
|---|---|---|---|---|
| Multi_modal_breast_cancer | 用原始 `tissue` 字段（breast/bone/liver/lung等） | **整个队列硬编码 'Metastatic'**，不分组织部位 | 研究设计上是，但按部位无关——患者层面"晚期转移性乳腺癌"研究，取样时全部已知远处转移 | 队列名就叫"metastatic breast cancer biopsies"；breast组织细胞也被无条件打上Metastatic |
| Wu_etal_2021_BRCA | 固定 'Breast' | 传参 `Primary_or_Metastatic='Primary'`，无任何覆盖，全员=Primary | 有：`Final_patient_stage = Stage`（Stage I/II/III/IV） | 代码保证100%细胞是'Primary'，但老`metastasis_label`按Stage把部分细胞（含14,341个Stage I!）标成Regional_Mets——Stage I不该有区域转移，明显是bug |
| 2102-Breastcancer (Qian et al 2020乳腺分支) | 固定 'Breast' | 传参 `Primary_or_Metastatic='Primary'`，无覆盖，全员=Primary | 有：`Final_patient_stage = TNM` | 同一篇论文的COAD/OVC分支都有M1覆盖，唯独BRCA分支没写，这正是审稿人举的"三个分支规则不一致"的例子 |
| GSE167036（原发+配对转移淋巴结） | 按`sample_type`：Tumor→Breast, Lymph Node→Lymph node | **整个队列硬编码 'Metastatic'**，不分组织 | 患者层面"配对原发-转移淋巴结"研究设计，是；但`Final_patient_stage='Unknown'`（无法读出具体分期） | 乳腺原发组织的细胞，只因为"这个患者也测了转移淋巴结"就被打成Metastatic |
| Single_cell_map_anti-PD1 (2 cohorts) | 固定 'Breast' | 传参默认/显式 `Primary_or_Metastatic='Primary'`，无覆盖，全员=Primary | 有：与临床表合并后得到 Stage II/III/Unknown | 代码保证全员Primary，老`metastasis_label`仍按Stage拆成Regional_Mets |
| GSE161529 | — | 未细查（数据量较小，未进入最终atlas主力癌种分布） | `Final_patient_stage='Unknown'` | 建议后续如需要可再查 |
| GSE225600 | — | 未细查 | — | 建议后续如需要可再查 |

## 结直肠癌 (Colorectal Cancer)

| Project_ID | 组织赋值 | Primary_or_Metastatic 旧规则 | 取样时是否已知远处转移可得？ | 备注 |
|---|---|---|---|---|
| 2098-Colorectalcancer (Qian et al 2020结直肠分支) | 固定 'Colon' | **TNM字符串含"M1"→'Metastatic'**（代码逐行覆盖） | 有：`Final_patient_stage = TNM` | 与OVC分支写法一致，与BRCA/LUCA分支不一致 |
| GSE225857（肝转移配对样本） | 按`organs`：CCT→Colon, LCT→Liver | **整个队列硬编码 'Metastatic'**，不分组织 | 患者层面"原发-肝转移配对"设计，是；无逐患者分期字段 | Colon组织(原发肠道)细胞因队列设计被打成Metastatic，与GSE167036乳腺案例是同一种bug模式 |
| GSE178341 | 固定 'Colon' | 先整体硬编码'Metastatic'，后被**"Metastasis stage (on resection specimen path report)"含"M1"→Metastatic，否则M0** 覆盖 | 有：切除标本病理报告里的转移分期，直接是取样时的M分期 | 这是四个数据集里对"取样时转移状态"记录最精确的一个 |
| GSE132465 | 固定 'Colon' | **"TNM stage"含"M1"→Metastatic，否则Primary**（逐细胞判断） | 有：`Final_patient_stage = Stage` | 与2098分支同类型规则 |

## 肺癌 (Lung Cancer)

| Project_ID | 组织赋值 | Primary_or_Metastatic 旧规则 | 取样时是否已知远处转移可得？ | 备注 |
|---|---|---|---|---|
| High-resolution_single-cell_atlas | 用原始`tissue`字段 | **混合规则**：tissue≠lung→Metastatic；否则若uicc_stage含"IV"→Metastatic；否则若stage=='III'→'Locally advanced'（后在Integrate步骤被并入'Primary'）；否则Primary | 有：`uicc_stage`（I/II/III/IV） | 同时使用部位和分期两种标准，且发明了第三类"Locally advanced"后来被强行归并 |
| HTAN_MSK_SCLC | 用原始`tissue`字段 | **混合规则**：tissue≠lung→Metastatic；elif Stage=='IV'→Metastatic（代码注释明确写"Primary tumor but from Stage IV (M1) patient"）；否则Primary | 有：`Stage at Dx` | 代码注释本身就承认这是在用患者M1状态标记原发灶细胞 |
| 2096-Lungcancer (Qian et al 2020肺分支) | 固定 'Lung' | 传参 `Primary_or_Metastatic='Primary'`，**无任何覆盖**，全员=Primary | 有：`Final_patient_stage = TNM` | 与BRCA分支一样没有M1覆盖，与COAD/OVC分支不一致 |

## 卵巢癌 (Ovarian Cancer)

| Project_ID | 组织赋值 | Primary_or_Metastatic 旧规则 | 取样时是否已知远处转移可得？ | 备注 |
|---|---|---|---|---|
| 2100-Ovariancancer (Qian et al 2020卵巢分支) | 固定 'Ovary' | **TNM字符串含"M1"→'Metastatic'**（与COAD分支写法完全一致） | 有：`Final_patient_stage = TNM` | 同一篇论文里COAD和OVC分支都覆盖了M1规则，BRCA/LUCA分支没有——四分支处理不一致的直接证据 |
| MSK_SPECTRUM | 用细粒度`author_tumor_supersite`（Adnexa/Bowel/Omentum/Peritoneum/Upper Quadrant/Other等） | **混合规则**：`supersite=='Adnexa' 且 stage不含'IV'`→Primary；否则（非Adnexa部位 **或** Stage IV）→Metastatic | 有：`gyn_diagnosis_figo_stage`（FIGO分期） | 这是本表里唯一同时、显式地把"部位"和"分期"结合在同一条if/else里的数据集；也是我们新规则里卵巢腹膜细分类别的主要来源 |
| GSE173682 | 固定 'Ovary' | **纯分期规则**：Stage以"IV"开头→Metastatic，否则Primary，与组织部位完全无关 | 有：`Stage` | Ovary组织的Stage IV患者细胞被老规则标成Metastatic，新规则下按部位规则改判No_Mets |
| Zhang_2019 | 固定 'Ovary' | 硬编码 'Primary'，无分期信息 | 无：`Final_patient_stage='Unknown'` | 最简单干净的一个，无冲突 |

---

## 小结：四个癌种、同一篇源论文（Qian et al. 2020）四个分支的规则对比

这是审稿人邮件里点名批评的具体例子，四个分支处理完全不一致：

| 分支 | Project_ID | 是否用了TNM M1覆盖规则？ |
|---|---|---|
| 乳腺 | 2102-Breastcancer | 否——全员固定'Primary' |
| 结直肠 | 2098-Colorectalcancer | **是** |
| 肺 | 2096-Lungcancer | 否——全员固定'Primary' |
| 卵巢 | 2100-Ovariancancer | **是** |

## "取样时是否已知远处转移"这个维度目前的可得性

除 GSE167036（乳腺，配对淋巴结研究，Unknown分期）、GSE225857（结直肠，肝转移配对，无逐患者分期）、
Zhang_2019（卵巢，Unknown分期）外，其余所有数据集都记录了取样时的患者分期（TNM/FIGO/uicc_stage等），
理论上可以重新提取一个独立的二值标记，例如：

```
primary_site_had_synchronous_distant_mets: {True, False, Unknown}
```

只对 `Final_tissue`=原发器官 的细胞有意义（非原发部位的细胞已经按活检部位规则单独判定 Regional/Distant，
不需要这个维度）。这个标记可以作为 `metastasis_label` 的补充列，而不必更改现有三分类的定义——
具体要不要采纳、以及采纳后怎么用（并入主标签 vs 作为二级分析/稳健性检验），需要你来定。
