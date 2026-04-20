"""Truth-model exports for both the pure Python model and the legacy Basilisk scaffold."""

from src.truth.attitude import AttitudeConfig, AttitudeHistory, AttitudeState, build_attitude_history, step_attitude_state
from src.truth.disturbances import (
    DisturbanceConfig,
    DisturbanceHistory,
    DisturbanceSample,
    build_disturbance_history,
    evaluate_disturbances,
)
from src.truth.environment import (
    EnvironmentConfig,
    EnvironmentHistory,
    EnvironmentSample,
    build_environment_history,
    evaluate_environment,
)
from src.truth.orbit import (
    OrbitConfig,
    OrbitHistory,
    OrbitState,
    OrbitTruthHistory,
    OrbitalElements,
    apply_orbit_truth_configuration,
    build_orbit_history,
    extract_orbit_truth_history,
    get_central_body_constants,
    orbital_elements_to_state,
    propagate_orbit,
    step_orbit_state,
)
from src.truth.truth_model import TruthModel, TruthModelResult

__all__ = [
    "AttitudeConfig",
    "AttitudeHistory",
    "AttitudeState",
    "DisturbanceConfig",
    "DisturbanceHistory",
    "DisturbanceSample",
    "EnvironmentConfig",
    "EnvironmentHistory",
    "EnvironmentSample",
    "OrbitConfig",
    "OrbitHistory",
    "OrbitState",
    "OrbitTruthHistory",
    "OrbitalElements",
    "TruthModel",
    "TruthModelResult",
    "apply_orbit_truth_configuration",
    "build_attitude_history",
    "build_disturbance_history",
    "build_environment_history",
    "build_orbit_history",
    "evaluate_disturbances",
    "evaluate_environment",
    "extract_orbit_truth_history",
    "get_central_body_constants",
    "orbital_elements_to_state",
    "propagate_orbit",
    "step_attitude_state",
    "step_orbit_state",
]
