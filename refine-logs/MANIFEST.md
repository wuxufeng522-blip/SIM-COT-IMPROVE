# Output Manifest

项目语言：中文。版本化文件是不可变审阅快照；固定文件是当前入口。

| Date | Versioned Artifact | Fixed Alias | Purpose | SHA-256 |
| --- | --- | --- | --- | --- |
| 2026-08-05 | `2026-08-05-reliability-gated-simcot-EXPERIMENT_PLAN.md` | `EXPERIMENT_PLAN.md` | claim-driven 实验与实施路线 | `F0F54A0B24D3ECDA26BAB364D55E0B2E42DCF4B1033832ABA2845E693E43848B` |
| 2026-08-05 | `2026-08-05-reliability-gated-simcot-EXPERIMENT_TRACKER.md` | `EXPERIMENT_TRACKER.md` | 运行顺序、门槛和状态追踪 | `099517343CFCB87BC7C20B5A6280D224C8970E856E0EAD14EDDDF157D08DC757` |

## Protocol Note

`experiment-plan` skill 引用的 `output-versioning.md`、`output-manifest.md` 和 `output-language.md` 在本机 skill 安装中缺失。本清单按 skill 正文可见要求实施安全降级：先创建日期版本，再生成固定入口，记录哈希，并沿用用户语言。
