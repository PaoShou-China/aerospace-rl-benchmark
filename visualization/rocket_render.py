"""Genesis-only display scene, synchronized cameras and streaming video."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .rocket_data import ROOT, MODELS, Episode, playback_times, rotation


def episode_digest(episode):
    return hashlib.sha256(json.dumps(episode.data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def overlay(main, close, values, episode, width, height, font_path=None):
    """CPU image composition, also usable without Genesis for verification."""
    from PIL import Image, ImageDraw, ImageFont
    def font(size):
        for p in (font_path, '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'):
            if p and Path(p).is_file():
                return ImageFont.truetype(str(p), size)
        return ImageFont.load_default()
    image = Image.new('RGB', (width, height), '#142532')
    main_width = width-300
    image.paste(Image.fromarray(np.asarray(main, np.uint8)).resize((main_width, height)), (0, 0))
    thumb = Image.fromarray(np.asarray(close, np.uint8)).resize((240, 180))
    image.paste(thumb, (main_width-252, height-196))
    draw = ImageDraw.Draw(image)
    draw.rectangle((main_width-252, height-218, main_width-12, height-196), fill='#142532')
    draw.text((main_width-246, height-215), 'PAD CAMERA / SAME TIME', font=font(12), fill='#e0f0f0')
    s, x, y = values['sample'], main_width+16, 18
    def line(label, value='', size=15):
        nonlocal y
        draw.text((x, y), label, font=font(12), fill='#a6bdc9'); y += 20
        if value:
            draw.text((x, y), value, font=font(size), fill='#edf6f8'); y += 28
    vec = lambda a: ' '.join(f'{v:7.2f}' for v in a)
    line('ROCKET LANDING', 'DEMO' if episode.data.get('demo') else 'RECORDED LOG')
    line('SIMULATION TIME', f"{s['t_s']:.3f} s")
    line('POSITION ENU [m]', vec(s['position_m']), 13)
    line('VELOCITY ENU [m/s]', vec(s['linear_velocity_mps']), 13)
    line('POSITION ERROR [m]', vec(values['position_error_m']), 13)
    line('HORIZONTAL / FOOT GAP [m]', f"{values['horizontal_error_m']:.2f} / {values['foot_height_m']:.2f}")
    line('FUEL USED [kg / %]', f"{values['fuel_used_kg']:.3f} / {values['fuel_used_percent']:.1f}")
    line('TOUCHDOWN ELAPSED [s]', f"{values['landing_elapsed_s']:.3f}" + (' LOCKED' if values['contact_time_s'] is not None else ' RUNNING'))
    terminal = values['terminal_elapsed_s']
    line('TERMINAL PHASE [s]', '--' if terminal is None else f'{terminal:.3f}')
    status = 'TERMINATED' if s['terminated'] else 'SETTLED' if s['complete'] else 'PAD CONTACT' if s['pad_contact'] else 'DESCENT'
    draw.rectangle((12, 12, 310, 44), fill='#142532')
    draw.text((22, 19), status, font=font(17), fill='#8fe1cd')
    return np.asarray(image)


class GenesisReplay:
    def __init__(self, episode, *, backend='cpu', width=1280, height=720, viewer=False):
        import genesis as gs
        self.gs, self.episode = gs, episode
        if width < 960 or height < 720 or width % 2 or height % 2:
            raise ValueError('Use even video dimensions, width >= 960 and height >= 720')
        if not gs._initialized:
            gs.init(backend=gs.cpu if backend == 'cpu' else gs.gpu, logging_level='warning')
        self.temp = tempfile.TemporaryDirectory(prefix='rocket-replay-')
        self.scene = None
        try:
            self._build(width, height, viewer)
        except BaseException:
            self.close()
            raise

    def _build(self, width, height, viewer):
        gs, m = self.gs, self.episode.metadata
        for rel, expected in m.get('assets_sha256', {}).items():
            # Logs cannot request an arbitrary external asset path.
            asset = (ROOT/rel).resolve()
            if not asset.is_relative_to(MODELS.resolve()) or hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Asset mismatch: {rel}; use the matching source/asset version')
        scene = self.scene = gs.Scene(show_viewer=viewer,
            sim_options=gs.options.SimOptions(gravity=(0, 0, 0)), renderer=gs.renderers.Rasterizer())
        def box(size, pos, color):
            return scene.add_entity(gs.morphs.Box(size=size, pos=pos, fixed=True, collision=False),
                                    surface=gs.surfaces.Plastic(color=color))
        box((200, 200, 10), (0, 0, -5), (.42, .51, .47, 1.))
        box((6, 150, .01), (22, 20, .01), (.31, .39, .38, 1.))
        for i in range(5):
            box((5, 8, 2.5), (-25-i*8, 30+(i%2)*9, 1.25), (.52, .60, .57, 1.))
        for i in range(10):
            box((.1, 4, .01), (22, -40+i*12, .025), (.8, .82, .72, 1.))
        def urdf(path, name, **kwargs):
            node = ET.fromstring(path.read_text().split('</robot>')[0]+'</robot>')
            flame = node.find("link[@name='flame']")
            if flame is not None:
                for v in list(flame.findall('visual')):
                    flame.remove(v)
            for mesh in node.findall('.//mesh'):
                mesh.set('filename', str((path.parent/mesh.get('filename')).resolve()))
            clean = Path(self.temp.name)/name
            ET.ElementTree(node).write(clean)
            return scene.add_entity(gs.morphs.URDF(file=str(clean), collision=False,
                align=False, merge_fixed_links=False, recompute_inertia=False, **kwargs))
        urdf(MODELS/'landing_pad.urdf', 'pad.urdf', pos=tuple(m['pad_position_m']), fixed=True)
        self.body = urdf(MODELS/'vehicles/rocket/rocket.urdf', 'rocket.urdf')
        for pos in ((-.35, 0, m['pad_surface_z_m']+.003), (.35, 0, m['pad_surface_z_m']+.003)):
            box((.10, .95, .004), pos, (.86, .88, .78, 1.))
        box((.75, .10, .004), (0, 0, m['pad_surface_z_m']+.003), (.86, .88, .78, 1.))
        self.plume = [scene.add_entity(gs.morphs.Cylinder(radius=.11-i*.012, height=.28,
            pos=(0, 0, -1e5), fixed=False, collision=False),
            surface=gs.surfaces.Plastic(color=(1., .7+i*.04, .24, .65))) for i in range(6)]
        self.markers = [scene.add_entity(gs.morphs.Sphere(radius=.055, pos=(0,0,-1e5), fixed=False, collision=False),
            surface=gs.surfaces.Plastic(color=(.12, .78, .7, 1.))) for _ in range(96)]
        self.trail_times = np.linspace(0, self.episode.duration, len(self.markers))
        self.trail_positions = [self.episode.at(t)['position_m'] for t in self.trail_times]
        self.follow = scene.add_camera(res=(width-300, height), pos=(20,-20,20), lookat=(0,0,3), fov=50, near=.03, far=3000)
        self.pad = scene.add_camera(res=(480,360), pos=(9,-12,5), lookat=(0,0,2.2), fov=45, near=.03, far=3000)
        scene.build()

    def images(self, time):
        s = self.episode.at(time)
        p, q = np.asarray(s['position_m']), np.asarray(s['quaternion_wxyz'])
        self.body.set_pos(tuple(p)); self.body.set_quat(tuple(q/np.linalg.norm(q)))
        focus = np.array((p[0]*.7, p[1]*.7, max(2.2, p[2]*.5)))
        distance = max(16., p[2]*1.4+6)
        direction = np.array((.65,-.75,.30)); direction /= np.linalg.norm(direction)
        self.follow.set_pose(pos=tuple(focus+distance*direction), lookat=tuple(focus))
        for marker, at, pos in zip(self.markers, self.trail_times, self.trail_positions):
            marker.set_pos(tuple(pos) if at <= time else (0,0,-1e5))
        gx, gy = s['gimbal_rad']
        direction = rotation(q) @ np.array((math.sin(gy), -math.sin(gx)*math.cos(gy), math.cos(gx)*math.cos(gy)))
        exhaust = -direction / np.linalg.norm(direction)
        axis = np.cross((0.,0.,1.), exhaust)
        w = 1+exhaust[2]
        orient = np.array((w,*axis)) if w > 1e-8 else np.array((0.,1.,0.,0.))
        orient /= np.linalg.norm(orient)
        nozzle = p + rotation(q) @ np.array((0,0,-2.25))
        visible = int(np.ceil(np.clip(s['throttle_actual'],0,1)*len(self.plume)))
        for i, segment in enumerate(self.plume):
            pos = nozzle+exhaust*(.14+i*.28)
            surface = self.episode.metadata['pad_surface_z_m'] if np.linalg.norm(pos[:2]) < self.episode.metadata['pad_radius_m'] else 0.
            show = i < visible and s['throttle_actual'] > .01 and pos[2] > surface+.15
            segment.set_pos(tuple(pos) if show else (0,0,-1e5)); segment.set_quat(tuple(orient))
        # Display only: deliberately NEVER call scene.step().
        return (self.follow.render(rgb=True, force_render=True)[0],
                self.pad.render(rgb=True, force_render=True)[0])

    def close(self):
        if self.scene is not None:
            self.scene.destroy(); self.scene = None
        self.temp.cleanup()


def render_video(episode, output, *, fps=30, speed=1., start=0., end=None,
                 backend='cpu', width=1280, height=720, font_path=None):
    import imageio.v2 as imageio
    output = Path(output)
    if output.exists() or output.with_suffix('.timeline.json').exists():
        raise FileExistsError(f'Output already exists: {output}')
    times = playback_times(episode.duration, fps, speed, start, end)
    output.parent.mkdir(parents=True, exist_ok=True)
    scene = GenesisReplay(episode, backend=backend, width=width, height=height)
    try:
        with imageio.get_writer(str(output), fps=fps, codec='libx264', quality=8, macro_block_size=1) as writer:
            for time in times:
                main, close = scene.images(float(time))
                writer.append_data(overlay(main, close, episode.metrics(float(time)), episode, width, height, font_path))
    finally:
        scene.close()
    timeline = dict(episode_sha256=episode_digest(episode), fps=fps, speed=speed, start_s=start,
                    end_s=episode.duration if end is None else end, frames=len(times), demo=episode.data.get('demo', None))
    output.with_suffix('.timeline.json').write_text(json.dumps(timeline, indent=2), encoding='utf-8')


def export_html(episode, video, output):
    """Share Genesis pixels + embedded metrics; no server, no remote assets."""
    import base64
    from html import escape
    video, output = Path(video), Path(output)
    if output.exists():
        raise FileExistsError(output)
    timeline = json.loads(video.with_suffix('.timeline.json').read_text())
    if timeline['episode_sha256'] != episode_digest(episode):
        raise ValueError('Video and trajectory do not belong to the same episode')
    times = playback_times(episode.duration, timeline['fps'], timeline['speed'], timeline['start_s'], timeline['end_s'])
    values = [episode.metrics(float(t)) for t in times]
    payload = json.dumps(dict(fps=timeline['fps'], values=values), separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    badge = '示意数据' if episode.data.get('demo') else 'Genesis 回放 · 记录数据'
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>火箭着陆回放</title><style>body{background:#142532;color:#eef5f8;font:16px/1.6 system-ui;margin:24px auto;padding:0 16px;max-width:1200px}video{width:100%;display:block}pre{white-space:pre-wrap;font-variant-numeric:tabular-nums}small{color:#b0c8d4}</style>
<h1>火箭着陆回放</h1><small>BADGE · 视频已包含同步双机位和 HUD</small>
<video controls preload="metadata" src="data:video/mp4;base64,VIDEO"></video><pre id="metrics" aria-live="off"></pre>
<script type="application/json" id="data">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent),v=document.querySelector('video'),o=document.getElementById('metrics');
const vec=a=>a.map(x=>x.toFixed(2)).join(', ');
function update(){const m=data.values[Math.min(data.values.length-1,Math.floor(v.currentTime*data.fps))],s=m.sample;
o.textContent='仿真时间 '+s.t_s.toFixed(3)+' s\\n位置 [m] '+vec(s.position_m)+'\\n速度 [m/s] '+vec(s.linear_velocity_mps)+'\\n位置偏差 [m] '+vec(m.position_error_m)+'\\n燃料消耗 '+m.fuel_used_kg.toFixed(3)+' kg / '+m.fuel_used_percent.toFixed(1)+'%\\n首次触地耗时 '+(m.contact_time_s===null?'未触地':m.contact_time_s.toFixed(3)+' s')+'\\n末端阶段耗时 '+(m.terminal_elapsed_s===null?'未开始':m.terminal_elapsed_s.toFixed(3)+' s');}
v.addEventListener('timeupdate',update);v.addEventListener('seeked',update);update();</script></html>'''
    page = page.replace('BADGE', escape(badge)).replace('VIDEO', base64.b64encode(video.read_bytes()).decode()).replace('PAYLOAD', payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
