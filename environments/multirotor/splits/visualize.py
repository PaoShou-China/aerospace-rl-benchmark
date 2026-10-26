"""Record and render one full-track checkpoint episode; no physics model changes."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(_ROOT))
from environments.multirotor.splits.evaluation import load_env_policy,record_episode


def render_frames(frames,result,geometry,out,fps=40):
    """Overview video: actual poses/trail, nominal gates, forward arrows and HUD.

    Drone marker/axis sizes are for visibility. Gate frames are front-face
    outlines rather than physics meshes; collision remains the analytic model.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio_ffmpeg
    if not frames or fps<1:raise ValueError('frames and positive fps required')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    corners=geometry.corners().cpu().numpy()
    centres=geometry.centres.cpu().numpy();normals=geometry.rotations[:,:,0].cpu().numpy()
    path=np.stack([f['p'] for f in frames])
    fig=plt.figure(figsize=(10,7.5),dpi=96);ax=fig.add_subplot(111,projection='3d')
    gate_lines=[]
    for i,(points,centre,normal) in enumerate(zip(corners,centres,normals)):
        closed=np.concatenate((points,points[:1]))
        line,=ax.plot(*closed.T,color='steelblue',linewidth=3);gate_lines.append(line)
        ax.quiver(*centre,*normal,length=.8,color='gray',arrow_length_ratio=.3)
        ax.text(*centre,str(i+1))
    points=np.concatenate((corners.reshape(-1,3),path));low=points.min(0)-1;high=points.max(0)+1
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(max(0,low[2]),high[2]),
           xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]')
    ax.set_box_aspect(high-low);ax.view_init(elev=25,azim=-55)
    trail,=ax.plot([],[],[],color='orange',linewidth=1.5)
    body,=ax.plot([],[],[],'ko',markersize=5)
    title=ax.set_title('')
    hud=fig.text(.02,.02,'',fontsize=9)
    with tempfile.TemporaryFile() as log:
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo',
                 '-pix_fmt','rgb24','-s','960x720','-r',str(fps),'-i','-',
                 '-an','-vcodec','libx264','-crf','18','-pix_fmt','yuv420p',str(out)]
        options=dict(stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        if sys.platform=='win32':options['creationflags']=subprocess.CREATE_NO_WINDOW
        try:
            with subprocess.Popen(command,**options) as encoder:
                try:
                    for i,frame in enumerate(frames):
                        trail.set_data_3d(*path[:i+1].T);body.set_data_3d(*path[i:i+1].T)
                        for gate,line in enumerate(gate_lines):
                            line.set_color('green' if gate==frame['target_id'] else 'steelblue')
                        title.set_text(f"Split-S | t={frame['time_s']:.3f}s | gates={frame['passed']} | "
                                       +(result if i==len(frames)-1 else f"target={frame['target_id']+1}"))
                        hud.set_text(f"speed={np.linalg.norm(frame['v']):.2f} m/s   "
                                     f"angular speed={np.linalg.norm(frame['w']):.2f} rad/s\n"
                                     'Arrows: forward direction | frame geometry is an approximation')
                        fig.canvas.draw()
                        rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                        encoder.stdin.write(rgb.tobytes())
                except BrokenPipeError:
                    pass
                finally:encoder.stdin.close()
                code=encoder.wait()
            if code:
                log.seek(0);raise RuntimeError(log.read().decode('utf-8',errors='replace'))
        finally:plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=_ROOT/'results'/'quadx_splits')
    parser.add_argument('--backend',choices=('cpu','gpu'),default='cpu')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--laps',type=int,default=1)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--video',type=Path)
    args=parser.parse_args()
    env,policy,_=load_env_policy(args.out,1,args.backend,args.seed,args.laps)
    try:
        frames,result=record_episode(env,policy,args.max_steps)
        video=args.video or args.out/'vis_splits.mp4'
        render_frames(frames,result,env.gate_geometry,video)
        np.savez_compressed(video.with_suffix('.npz'),**{k:np.stack([f[k] for f in frames]) for k in frames[0]})
        video.with_suffix('.json').write_text(json.dumps(dict(result=result,frames=len(frames),
            seed=args.seed,task=env.cfg,checkpoint=str(args.out/'checkpoint.pt')),indent=2),encoding='utf-8')
        print(f'{result}: {video}',flush=True)
    finally:env.close()


if __name__=='__main__':main()
