"""Create a reviewable table and figure from saved generation evaluations."""
import json
from run_evaluation_experiment import HERE
from evaluate_saved_generations import OUT, SWEEP, summarize


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = json.loads((SWEEP/'summary.json').read_text(encoding='utf-8'))
    status = json.loads((SWEEP/'status.json').read_text(encoding='utf-8'))
    reference = summarize(HERE/'experiments/lookahead5_resume_2000/evaluations/continued_1201.json')
    baseline = summarize(OUT/'baseline_2000.json')
    lines = ['# Split-S 2000–5000 代检查点评估', '',
        f'状态：{status["phase"]}，完成 {len(rows)}/{status["total"]} 个检查点。', '',
        '每个模型分别测试随机连续四门和固定起点完整七门；每项使用种子 7、17、27，各 64 个环境，共 192 个首回合。'
        '使用确定性推理，每回合上限 20 秒，四门测试不启用课程状态缓存。'
        '七门测试从第一门前的固定中点开始，不是原始地面起点。', '',
        '文件名为零基更新编号：model_2500.pt 对应已完成 2501 次更新；model_4999.pt 对应完成 5000 次更新。'
        '完赛耗时只对成功回合求平均；低成功率模型的该指标有选择偏差，不能单独用于比较速度。'
        '固定七门的种子重复不代表 192 种不同起点。', '',
        '已有评估结果在环境和评估代码一致时复用，其余检查点补测。详见 manifest.json、summary.json 和逐回合 JSON。', '',
        '| 模型文件 | 已完成更新 | 四门成功数 / 192 | 四门成功耗时(s) | 七门成功数 / 192 | 七门平均通过数 | 七门成功耗时(s) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def seconds(value):
        return '—' if value is None else f'{value:.3f}'
    for row in rows:
        a = row['protocols']['random_four_gates']; b = row['protocols']['fixed_full_track']
        lines.append(f'| {row["checkpoint"]} | {row["completed_updates"]} | {a["successes"]} | '
            f'{seconds(a["mean_success_time_s"])} | {b["successes"]} | {b["gates_passed"]:.4f} | '
            f'{seconds(b["mean_success_time_s"])} |')
    best = max(rows, key=lambda r:(r['protocols']['fixed_full_track']['success'],
                                  r['protocols']['fixed_full_track']['gates_passed']))
    lines += ['', '## 对照与失败位置', '',
        f'这次 2000–5000 代范围内，七门成功率最高的检查点为 {best["checkpoint"]}：'
        f'{best["protocols"]["fixed_full_track"]["successes"]}/192。',
        f'此前 model_1200.pt 的七门结果为 {reference["fixed_full_track"]["successes"]}/192，'
        f'成功耗时 {seconds(reference["fixed_full_track"]["mean_success_time_s"])} 秒。',
        f'续训前 2000 次更新的四门成功耗时为 {seconds(baseline["random_four_gates"]["mean_success_time_s"])} 秒。', '',
        '| 模型 | 七门失败时正在接近的门：次数 | 门框碰撞 | 触地（无门框碰撞） | 越界 | 超时 |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        p = row['protocols']['fixed_full_track']
        gates = ', '.join(f'{k}: {v}' for k,v in sorted(p['failures_by_target_gate'].items())) or '无'
        lines.append(f'| {row["checkpoint"]} | {gates} | {p["gate_collision"]:.1%} | '
            f'{p["ground_contact"]:.1%} | {p["out_of_bounds"]:.1%} | {p["timeout"]:.1%} |')
    lines += ['', '失败门编号表示失败时的目标门，不保证是实际碰撞的门；碰撞和越界标记可能重叠。', '',
        '训练日志也记录了 3300 代附近的下降：完成 3201、3301、3401 次更新时，'
        '最近 1024 个训练回合的成功率分别为 98.24%、60.74%、95.80%。'
        '来源：../training_metrics/3201.json、3301.json、3401.json。'
        '训练窗口包含课程缓存初始化，与这里的独立四门测试分布不同，数值不应直接等同。', '',
        '## 设定差异', '',
        '四门训练在接近第四门时仍观察第五门，过第四门就结束；七门测试在接近第七门时第二门的观测置零。'
        '因此即使四门成功率接近 100%，也不能推断七门能稳定完赛。末门观测差异是待验证原因，'
        '本次检查点评估没有改变观测规则，不能单凭失败位置确定因果。', '',
        '![Evaluation curves](generation_comparison.png)', '']
    (SWEEP/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    x = [r['completed_updates'] for r in rows]
    fig, axes = plt.subplots(3,1,figsize=(11,10),sharex=True,layout='constrained')
    for name,label in [('random_four_gates','4-gate segments'),('fixed_full_track','Full 7-gate track')]:
        axes[0].plot(x,[100*r['protocols'][name]['success'] for r in rows],'.-',label=label)
    axes[0].set(ylabel='Success (%)',ylim=(-3,103));axes[0].legend()
    axes[1].plot(x,[r['protocols']['random_four_gates']['mean_success_time_s'] for r in rows],'.-')
    axes[1].set(ylabel='4-gate success time (s)')
    axes[2].plot(x,[r['protocols']['fixed_full_track']['gates_passed'] for r in rows],'.-',color='tab:orange')
    axes[2].set(ylabel='Full-track gates passed',xlabel='Completed PPO updates',ylim=(0,7.1))
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle(f'Split-S checkpoint evaluation: {len(rows)}/{status["total"]} checkpoints, 192 episodes/protocol')
    fig.savefig(SWEEP/'generation_comparison.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
