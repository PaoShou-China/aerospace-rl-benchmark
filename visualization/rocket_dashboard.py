"""Evidence-first data model for the offline rocket replay dashboard.

The dashboard deliberately reports a *single episode*.  It never upgrades one
trajectory into an algorithm-level claim.  Comparisons are accepted only when
the recorded task, seed, configuration and initial state match.
"""
from __future__ import annotations

import json
import math
from typing import Iterable

import numpy as np

from .rocket_data import Episode, rotation


METHOD_STYLES = (
    ('#56B4E9', '', 'circle'),
    ('#E69F00', '12 7', 'square'),
    ('#CC79A7', '3 6', 'diamond'),
    ('#009E73', '12 5 3 5', 'triangle'),
    ('#F0E442', '2 5', 'cross'),
)

FAILURE_LABELS = {
    'hard_pad_speed': '触地速度超过硬碰撞门槛',
    'hard_pad_spin': '触地角速度超过硬碰撞门槛',
    'ground_contact': '发生非着陆台接触',
    'out_of_bounds': '飞出任务边界',
    'time_limit': '达到任务时限',
}


def method_label(episode: Episode) -> str:
    """Return a human-facing label without inventing an algorithm identity."""
    metadata = episode.metadata
    if metadata.get('method_label'):
        return str(metadata['method_label'])
    source = str(metadata.get('source', ''))
    if source == 'benchmark_pdg_baseline':
        return 'PDG 基线'
    if source == 'benchmark_checkpoint':
        config = metadata.get('requested_config') or {}
        algo = str(config.get('algo', '')).strip()
        return {'ppo': 'PPO', 'flash_sac': 'FlashSAC'}.get(algo, algo or '学习策略')
    if episode.data.get('demo'):
        return '示意回合'
    return source.replace('_', ' ').strip() or '记录回合'


def _outcome(episode: Episode) -> tuple[str, str]:
    result = episode.data.get('result') or {}
    last = episode.data['samples'][-1]
    complete = bool(result.get('complete', last.get('complete', False)))
    terminated = bool(result.get('terminated', last.get('terminated', False)))
    truncated = bool(result.get('truncated', last.get('truncated', False)))
    if bool(result.get('collection_limited', False)):
        return 'incomplete', '采集未完成'
    if complete:
        return 'success', '满足着陆判据'
    if terminated or truncated:
        return 'failure', '未满足着陆判据'
    return 'unknown', '结果不可判定'


def _touch_values(episode: Episode) -> tuple[float | None, float | None, float | None]:
    gate = next((event for event in episode.data.get('events', [])
                 if event.get('type') == 'pad_contact_gate'), None)
    if gate is not None:
        return (gate.get('gate_speed_mps'), gate.get('gate_spin_radps'), gate.get('tilt_rad'))
    contact = episode.event_time('pad_contact')
    if contact is None:
        return None, None, None
    sample = episode.at(contact)
    speed = float(np.linalg.norm(sample['linear_velocity_mps']))
    spin = float(np.linalg.norm(sample['angular_velocity_radps']))
    matrix = rotation(sample['quaternion_wxyz'])
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1., 1.)))
    return speed, spin, math.hypot(roll, pitch)


def episode_summary(episode: Episode) -> dict:
    """Extract evidence displayed in the result card and comparison table."""
    contact = episode.event_time('pad_contact', 'ground_contact')
    at_contact = episode.metrics(contact if contact is not None else episode.duration)
    at_end = episode.metrics(episode.duration)
    speed, spin, tilt = _touch_values(episode)
    outcome, outcome_label = _outcome(episode)
    return {
        'outcome': outcome,
        'outcome_label': outcome_label,
        'horizontal_error_m': at_contact['horizontal_error_m'],
        'error_3d_m': at_contact['error_3d_m'],
        'fuel_used_kg': at_end['fuel_used_kg'],
        'fuel_used_percent': at_end['fuel_used_percent'],
        # The result card says "first pad contact".  Do not substitute episode
        # duration when no contact exists; that would turn a failure into a
        # misleading touchdown time.
        'landing_elapsed_s': contact,
        'touch_speed_mps': speed,
        'touch_spin_radps': spin,
        'touch_tilt_rad': tilt,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def ensure_comparable(primary: Episode, comparison: Episode) -> None:
    """Reject visually persuasive but experimentally mismatched comparisons."""
    a, b = primary.metadata, comparison.metadata
    checks = (
        ('任务', a.get('task'), b.get('task')),
        ('随机种子', a.get('seed'), b.get('seed')),
        ('环境配置', _canonical(a.get('effective_env_config')), _canonical(b.get('effective_env_config'))),
        ('目标位置', _canonical(a.get('target_position_m')), _canonical(b.get('target_position_m'))),
    )
    for label, left, right in checks:
        if left != right:
            raise ValueError(f'Cannot compare episodes with different {label}')
    left, right = primary.data['samples'][0], comparison.data['samples'][0]
    for key in ('position_m', 'quaternion_wxyz', 'linear_velocity_mps', 'angular_velocity_radps'):
        if not np.allclose(left[key], right[key], rtol=0., atol=1e-9):
            raise ValueError(f'Cannot compare episodes with different initial {key}')


def _trajectory(episode: Episode, limit: int = 320) -> list[dict]:
    samples = episode.data['samples']
    if len(samples) <= limit:
        selected = samples
    else:
        indices = np.linspace(0, len(samples) - 1, limit).round().astype(int)
        selected = [samples[index] for index in np.unique(indices)]
    return [{'t_s': sample['t_s'], 'position_m': sample['position_m']} for sample in selected]


def _fmt(value, unit, digits=2, missing='未记录'):
    return missing if value is None else f'{value:.{digits}f} {unit}'.strip()


def _conclusion(primary: Episode, methods: list[dict]) -> dict:
    current = methods[0]
    summary = current['summary']
    demo = bool(primary.data.get('demo'))
    result = primary.data.get('result') or {}
    if demo:
        title = '示意回合已播放完成'
        body = '界面与指标链路可用于检查交互，但这些数值不是一次真实仿真实验。'
    elif summary['outcome'] == 'success':
        title = '本回合满足着陆判据'
        body = (f"{current['label']} 在本次初始条件下完成着陆；"
                f"首次触垫水平误差 {summary['horizontal_error_m']:.2f} m，"
                f"燃料消耗 {summary['fuel_used_kg']:.2f} kg。")
    elif summary['outcome'] == 'incomplete':
        title = '本回合证据不足'
        body = '轨迹采集在任务结束前停止，不能据此判断方法成功或失败。'
    elif summary['outcome'] == 'failure':
        reasons = [FAILURE_LABELS.get(reason, reason) for reason in result.get('failure_reasons', [])]
        suffix = f"；主要记录：{'、'.join(reasons)}" if reasons else ''
        title = '本回合未满足着陆判据'
        body = (f"{current['label']} 在本次初始条件下未完成有效着陆{suffix}。"
                f"首次触垫水平误差 {summary['horizontal_error_m']:.2f} m。")
    else:
        title = '本回合结果不可判定'
        body = '日志没有完整的任务结束标记，界面仅展示已记录的轨迹与遥测。'

    baseline = next((method for method in methods[1:] if method['role'] == 'baseline'), None)
    if baseline is not None and not demo:
        other = baseline['summary']
        if summary['outcome'] == 'success' and other['outcome'] == 'failure':
            body += f" 同条件下，{baseline['label']} 本回合未满足判据。"
        elif summary['outcome'] == 'failure' and other['outcome'] == 'success':
            body += f" 同条件下，{baseline['label']} 本回合满足判据。"
        else:
            precision_delta = other['horizontal_error_m'] - summary['horizontal_error_m']
            fuel_delta = summary['fuel_used_kg'] - other['fuel_used_kg']
            clauses = []
            if abs(precision_delta) >= .005:
                direction = '减少' if precision_delta > 0 else '增加'
                clauses.append(f'水平误差较 {baseline["label"]}{direction} {abs(precision_delta):.2f} m')
            if abs(fuel_delta) >= .005:
                direction = '增加' if fuel_delta > 0 else '减少'
                clauses.append(f'燃料消耗{direction} {abs(fuel_delta):.2f} kg')
            if clauses:
                body += ' 同条件单回合对比：' + '，'.join(clauses) + '。'

    evidence = [
        {'label': '任务判定', 'value': summary['outcome_label'],
         'detail': '环境原始 complete / termination 标记',
         'tone': 'good' if summary['outcome'] == 'success' else 'bad' if summary['outcome'] == 'failure' else 'warn'},
        {'label': '首次触垫水平误差', 'value': _fmt(summary['horizontal_error_m'], 'm'),
         'detail': '绝对值，不用综合分数替代', 'tone': 'neutral'},
        {'label': '首次触垫耗时', 'value': _fmt(summary['landing_elapsed_s'], 's', 2, '未触垫'),
         'detail': '从有效日志起点计时', 'tone': 'neutral'},
        {'label': '燃料消耗', 'value': _fmt(summary['fuel_used_kg'], 'kg'),
         'detail': f"本回合初始燃料的 {summary['fuel_used_percent']:.1f}%", 'tone': 'neutral'},
    ]
    caveat = ('示意数据，不能作为实验结论。' if demo else
              '仅比较同一初始条件下的单次回合；不能替代多随机种子、场景覆盖和置信区间统计。'
              if len(methods) > 1 else
              '仅代表本次回合；不能据此判断该方法的总体有效性。')
    return {'eyebrow': '示意结果' if demo else '本回合表现', 'title': title,
            'body': body, 'evidence': evidence, 'caveat': caveat}


def build_dashboard(primary: Episode, comparisons: Iterable[Episode] = ()) -> dict:
    episodes = [primary, *list(comparisons)]
    if len(episodes) > 5:
        raise ValueError('The dashboard supports at most five methods at once')
    for episode in episodes[1:]:
        ensure_comparable(primary, episode)

    labels: dict[str, int] = {}
    methods = []
    for index, episode in enumerate(episodes):
        label = method_label(episode)
        labels[label] = labels.get(label, 0) + 1
        shown_label = label if labels[label] == 1 else f'{label} #{labels[label]}'
        source = str(episode.metadata.get('source', ''))
        role = 'baseline' if 'baseline' in source else ('primary' if index == 0 else 'comparison')
        summary = episode_summary(episode)
        color, dash, marker = METHOD_STYLES[index]
        methods.append({
            'id': f'method-{index + 1}', 'label': shown_label, 'role': role,
            'color': color, 'dash': dash, 'marker': marker,
            'outcome': summary['outcome'], 'outcome_label': summary['outcome_label'],
            'summary': summary, 'trajectory': _trajectory(episode),
        })

    return {
        'demo': bool(primary.data.get('demo')), 'primary_id': methods[0]['id'],
        'events': primary.data.get('events', []), 'methods': methods,
        'conclusion': _conclusion(primary, methods),
    }
