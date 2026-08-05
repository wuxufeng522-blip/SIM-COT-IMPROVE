# Output Manifest

项目语言：中文。版本化文件是不可变审阅快照；固定文件是当前入口。

| Date | Versioned Artifact | Fixed Alias | Purpose | SHA-256 |
| --- | --- | --- | --- | --- |
| 2026-08-05 | `2026-08-05-reliability-gated-simcot-EXPERIMENT_PLAN.md` | `EXPERIMENT_PLAN.md` | claim-driven 实验与实施路线 | `F0F54A0B24D3ECDA26BAB364D55E0B2E42DCF4B1033832ABA2845E693E43848B` |
| 2026-08-05 | `2026-08-05-reliability-gated-simcot-EXPERIMENT_TRACKER.md` | `EXPERIMENT_TRACKER.md` | 运行顺序、门槛和状态追踪 | `099517343CFCB87BC7C20B5A6280D224C8970E856E0EAD14EDDDF157D08DC757` |
| 2026-08-05 | `../idea-stage/docs/2026-08-05-research_contract.md` | `../idea-stage/docs/research_contract.md` | 冻结研究主张、门槛与变更控制 | `4266769738612E95259B843E865E4BA28CA48432F3E70E4E0AE136B361718242` |
| 2026-08-05 | `2026-08-05-m0-EXPERIMENT_TRACKER.md` | `EXPERIMENT_TRACKER.md` | M0 完成状态与下一队列 | `876116DB6FB29609D3E9C2E1515A092473419CE148070891A61DDA9B6D53E706` |
| 2026-08-05 | `2026-08-05-m0-EXPERIMENT_RESULTS.md` | `EXPERIMENT_RESULTS.md` | 官方复现、单卡预检与审计结论 | `4EF53DD36328DDFAB6642375F5169B196DF1533B49B8AE0ABC4CDDB8B2EB4C6A` |

## Protocol Note

`experiment-plan` skill 引用的 `output-versioning.md`、`output-manifest.md` 和 `output-language.md` 在本机 skill 安装中缺失。本清单按 skill 正文可见要求实施安全降级：先创建日期版本，再生成固定入口，记录哈希，并沿用用户语言。
