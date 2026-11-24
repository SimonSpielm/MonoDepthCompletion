"""Configuration and URDF paths shared by EDAN and MAYA collision utilities.

Conventions
-----------
* EDAN: 8 reduced DOF = 1 additional axis + 7 arm joints.
* MAYA: 7 reduced DOF = 7 arm joints.
* Generic7R and SMMPL always operate on the LAST seven joints.
* The optional spherized URDF is not used by default for collision checks.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MARGIN = 0.005
DEFAULT_ENV_GEOM_NAME = "octree_env"


@dataclass(frozen=True)
class RobotCollisionConfig:
    """Reduced-model settings. Extra joints, if any, precede the seven arm joints."""

    robot_name: str
    dof: int
    ee_frame: str
    elbow_joint: str
    active_vmask_full: np.ndarray
    margin: float
    env_geom_name: str
    robot_collision_geom_names: frozenset[str]
    urdf_generic7r_path: str
    urdf_path: str
    spherized_urdf: str | None
    # Joints locked by the frocc backend (base + gripper driver); <mimic>
    # fingers follow their driver automatically and are not listed here.
    frocc_locked_joints: dict

    @property
    def extra_dof(self) -> int:
        return self.dof - 7


# The active velocity masks refer to the original, UNREDUCED Pinocchio models.
_ROBOT_PARAMS = {
    "edan": {
        "generic7r_filename": "lbr_plane.urdf",
        "spherized_filename": "edan_spherized.urdf",
        "ee_frame": "right_arm",
        "elbow_joint": "lwr_joint_4",
        "dof": 8,
        "active_vmask_full": np.array(
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            dtype=int,
        ),
        "robot_collision_geom_names": {
            "lwr_link_3_0", "lwr_link_4_0", "lwr_link_5_0", "lwr_link_6_0",
            "lwr_link_7_0", "robotiq_85_base_link_0", "left_outer_finger_0",
            "right_outer_finger_0", "left_inner_finger_0", "right_inner_finger_0",
        },
        "frocc_locked_joints": {
            "base_link_x": 0.0, "base_link_y": 0.0, "base_link_yaw": 0.0,
            "finger_joint": 0.0,
        },
    },
    "maya": {
        "generic7r_filename": "maya_only_arm.urdf",
        "spherized_filename": None,
        "ee_frame": "right_arm",
        "elbow_joint": "Connection_28",
        "dof": 7,
        "active_vmask_full": np.array(
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=int
        ),
        "robot_collision_geom_names": {
            "Link_2_0", "Link_3_0", "Link_4_0", "Link_5_0", 
            "Link_6_0",
            "robotiq_85_base_link_0", "left_outer_finger_0", "right_outer_finger_0",
            "left_inner_finger_0", "right_inner_finger_0",
        },
        "frocc_locked_joints": {
            "base_link_x": 0.0, "base_link_y": 0.0, "base_link_yaw": 0.0,
            "finger_joint": 0.0,
        },
    },
}


def _require_robot_name() -> str:
    name = os.environ.get("ROBOT_NAME")
    if name not in _ROBOT_PARAMS:
        raise EnvironmentError(
            f"Set ROBOT_NAME to one of {sorted(_ROBOT_PARAMS)}; got {name!r}"
        )
    return name


def _require_main() -> Path:
    main = os.environ.get("MAIN")
    if not main:
        raise EnvironmentError("Set MAIN to the project root directory")
    return Path(main).expanduser().resolve()


def _urdf_dir(robot_name: str, main: Path) -> Path:
    return main / "odb" / robot_name / "urdf_model" / "urdf"


def _require_file(path: Path, description: str) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return str(path)


def get_generic7r_urdf_path() -> str:
    name = _require_robot_name()
    return _require_file(
        _urdf_dir(name, _require_main()) / _ROBOT_PARAMS[name]["generic7r_filename"],
        "Generic7R URDF",
    )


def get_urdf_path() -> str:
    """Materialize the robot's templated main URDF into a temporary file."""
    name = _require_robot_name()
    template = _urdf_dir(name, _require_main()) / f"{name}_arm.urdf.template"
    _require_file(template, "URDF template")

    model_path = template.parent.parent  # .../urdf_model
    xml = (
        template.read_text(encoding="utf-8")
        .replace("<MODEL_PATH>", str(model_path))
        .replace('version="0.1"', 'version="1.0"')
    )

    fd, filename = tempfile.mkstemp(prefix=f"{name}_", suffix=".urdf")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(xml)
    except BaseException:
        Path(filename).unlink(missing_ok=True)
        raise
    return str(Path(filename).resolve())


def get_spherized_urdf_path(robot_name: str, main: Path) -> str | None:
    """EDAN has a spherized URDF; MAYA has none and returns None."""
    filename = _ROBOT_PARAMS[robot_name]["spherized_filename"]
    if filename is None:
        return None
    return _require_file(_urdf_dir(robot_name, main) / filename, "Spherized URDF")


def get_robot_collision_config() -> RobotCollisionConfig:
    """Build one configuration for the robot selected via ROBOT_NAME and MAIN."""
    name = _require_robot_name()
    main = _require_main()
    params = _ROBOT_PARAMS[name]
    dof = params["dof"]
    if dof < 7 or int(params["active_vmask_full"].sum()) != dof:
        raise ValueError(f"Inconsistent DOF / active velocity mask for {name}")

    return RobotCollisionConfig(
        robot_name=name,
        dof=dof,
        ee_frame=params["ee_frame"],
        elbow_joint=params["elbow_joint"],
        active_vmask_full=params["active_vmask_full"].copy(),
        margin=DEFAULT_MARGIN,
        env_geom_name=DEFAULT_ENV_GEOM_NAME,
        robot_collision_geom_names=frozenset(params["robot_collision_geom_names"]),
        urdf_generic7r_path=get_generic7r_urdf_path(),
        urdf_path=get_urdf_path(),
        spherized_urdf=get_spherized_urdf_path(name, main),
        frocc_locked_joints=dict(params["frocc_locked_joints"]),
    )
