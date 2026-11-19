"""State-only Phi-Rocket contracts, units and complete parameter registry.

All experiments use this registry and all operators. A parameter which is not
observable retains uncertainty; it is never removed from the physical model.
Coefficients are normalized coordinates in [-1,1], not independent measurements
of true masses, efficiencies or contact constants.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch


# name, normalized-coordinate scale, physical parameterization, SI unit/group
PARAMETERS = (
 ('thrust_per_mass_gain',.7,'log','flight'),
 ('drag_x_gain',.7,'log','aero'),('drag_y_gain',.7,'log','aero'),('drag_z_gain',.7,'log','aero'),
 ('transverse_inertia_gain',.7,'log','inertia'),('axial_inertia_gain',.7,'log','inertia'),
 ('fin_aerodynamic_gain',.7,'log','aero'),('engine_lag_gain',.7,'log','actuator'),
 ('gimbal_lag_gain',.7,'log','actuator'),('contact_stiffness_gain',1.2,'log','contact'),
 ('contact_damping_gain',1.2,'log','contact'),('friction_gain',1.2,'log','contact'),
 ('dry_mass_gain',.3,'log','mass'),('fuel_mass_gain',.3,'log','mass'),
 ('fuel_burn_gain',.3,'log','fuel'),
 ('com_x_offset',.08,'linear','m'),('com_y_offset',.08,'linear','m'),('com_z_offset',.15,'linear','m'),
 ('inertia_y_gain',.3,'log','inertia'),('inertia_xy_coupling',.15,'linear','inertia'),
 ('inertia_xz_coupling',.15,'linear','inertia'),('inertia_yz_coupling',.15,'linear','inertia'),
 ('fin_response_gain',.5,'log','actuator'),('fin_common_bias',.08,'linear','normalized command'),
 ('fin_lift_slope_gain',.3,'log','aero'),('fin_drag_gain',.3,'log','aero'),
 ('fin_moment_gain',.3,'log','aero'),
 ('gimbal_x_bias',.01,'linear','rad'),('gimbal_y_bias',.01,'linear','rad'),
 ('throttle_bias',.05,'linear','normalized throttle'),
 ('thrust_axis_x',.015,'linear','rad'),('thrust_axis_y',.015,'linear','rad'),
 ('wind_world_x',8.,'linear','m/s'),('wind_world_y',8.,'linear','m/s'),('wind_world_z',3.,'linear','m/s'),
 ('air_density_gain',.2,'log','aero'),('restitution_gain',.7,'log','contact'),
 ('static_friction_gain',.4,'log','contact'),('leg_height_offset',.02,'linear','m'),
 ('pad_height_offset',.02,'linear','m'),('pad_friction_gain',.3,'log','contact'),
 ('position_bias_x',.05,'linear','m'),('position_bias_y',.05,'linear','m'),('position_bias_z',.05,'linear','m'),
 ('velocity_bias_x',.05,'linear','m/s'),('velocity_bias_y',.05,'linear','m/s'),('velocity_bias_z',.05,'linear','m/s'),
 ('gyro_bias_x',.01,'linear','rad/s'),('gyro_bias_y',.01,'linear','rad/s'),('gyro_bias_z',.01,'linear','rad/s'),
 ('attitude_bias_x',.01,'linear','rad'),('attitude_bias_y',.01,'linear','rad'),('attitude_bias_z',.01,'linear','rad'),
 ('fin_roll_bias',.04,'linear','normalized command'),('fin_pitch_bias',.04,'linear','normalized command'),
 ('fin_yaw_bias',.04,'linear','normalized command'),('ground_height_offset',.01,'linear','m'),
 ('gravity_gain',.01,'log','gravity'),('contact_tangent_compliance_gain',.5,'log','contact'),
 ('contact_rolling_damping_gain',.5,'log','contact'),('shutdown_lag_gain',.5,'log','actuator'),
)
NP=len(PARAMETERS)
NX=20                         # 21 physical coordinates, quaternion uses 3D errors
ND=6
NJ=NX+NP+ND
STATE_SCALE=(.1,.1,.1,.3,.3,.3,.03,.03,.03,.1,.1,.1,.05,.05,.05,.05,.05,.05,.05,.002)
WRENCH_BOUND=(50.,50.,100.,20.,20.,10.)


@dataclass(frozen=True)
class RocketTask:
    target_position: tuple=(0.,0.,2.575)
    target_quaternion: tuple=(0.,0.,0.,1.)
    target_velocity: tuple=(0.,0.,0.)
    pad_center: tuple=(0.,0.)
    pad_radius: float=2.
    pad_height: float=.15
    mission_seconds: float=30.
    fuel_weight: float=.02
    smooth_weight: float=.02
    risk_weight: float=.2
    stable_seconds: float=2.
    max_touchdown_speed: float=1.
    max_touchdown_angular_speed: float=.3
    support_tilt: float=.1

    def __post_init__(self):
        if not 0<self.max_touchdown_speed<=1. or not 0<self.max_touchdown_angular_speed<=.3 or not 0<self.support_tilt<=.1:
            raise ValueError('task preferences cannot relax engineering touchdown/attitude limits')
        if self.stable_seconds<2 or self.mission_seconds<=0 or self.pad_radius<=0:
            raise ValueError('invalid task duration or platform geometry')
        if min(self.fuel_weight,self.smooth_weight,self.risk_weight)<0:
            raise ValueError('task cost weights must be nonnegative')
        values=(*self.target_position,*self.target_quaternion,*self.target_velocity,*self.pad_center)
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite task geometry')
        if abs(sum(v*v for v in self.target_quaternion)-1)>1e-4:
            raise ValueError('target attitude must be a unit xyzw quaternion')

    def tensor(self,like):
        values=(*self.target_position,*self.target_quaternion,*self.target_velocity,
            *self.pad_center,self.pad_radius,self.pad_height,self.mission_seconds,
            self.fuel_weight,self.smooth_weight,self.risk_weight,self.stable_seconds,
            self.max_touchdown_speed,self.max_touchdown_angular_speed,self.support_tilt)
        return like.new_tensor(values).expand(len(like),-1)


@dataclass(frozen=True)
class EngineeringEnvelope:
    """An assumption contract, not a claim that an error bound is calibrated."""
    control_dt: float=.025
    measurement_max_gap: float=.1
    parameter_drift_per_second: float=.002
    normalized_process_error: float=.002
    model_error_bound_validated: bool=False
    numerical_error_bound_validated: bool=False
    model_error_tangent_per_second: tuple=(0.,)*20
    numerical_error_tangent_per_step: tuple=(0.,)*20
    control_wall_budget_seconds: float=20.
    minimum_reserve_seconds: float=.1

    def describe(self):return asdict(self)


class BeliefLayout:
    def __init__(self,plan_steps=8):
        i=0
        for name,width in [('hidden',128),('state',21),('theta',NP),('disturbance',6),
            ('covariance',NJ*NJ),('lower',NJ),('upper',NJ),('constraint_matrix',NX*NJ),
            ('constraint_radius',NX),('constraint_center',NJ),('modes',5),
            ('invalid',1),('age',1),('clock',1),('plan',plan_steps*7),('error_ema',NX),('contact_age',1),
            ('evidence_window',4*50),('evidence_valid',4),('version',1),
            ('learned_theta_offset',NP),('learned_disturbance_offset',ND)]:
            setattr(self,name,slice(i,i+width));i+=width
        self.width=i


def parameter_manifest():
    return [dict(index=i,name=n,scale=s,transform=t,unit_or_group=g) for i,(n,s,t,g) in enumerate(PARAMETERS)]
