# 仓库维护记录

- 2026-07-19: 执行 git filter-repo 重写历史，移除所有 *.npy 缓存及 all17/all17_split 数据集；
  仓库本地 .git 由 ~2GB 降至 182MB。随后在 gitee 网页执行 Repository GC。
- 本文件用于触发一次空提交式 push，验证 gitee 配额/体积是否已恢复。
