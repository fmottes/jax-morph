import jax_morph as jxm

EXPECTED = {
    '__version__',
    'build_state_from_model',
    'Model',
    'SimulationStep',
    'StochasticStep',
    'StepType',
    'simulate',
    'trajectory_logp',
    'transition_logp',
    'StateFieldSpec',
    'BaseState',
    'check_stochastic_step',
    'save_model',
    'load_model',
    'save_state',
    'load_state',
    'save_trajectory',
    'load_trajectory',
    'TrajectoryRecord',
    'POSITION',
    'RADIUS',
    'CELLTYPE',
    'ALIVE',
    'TIME',
    'ad_utils',
    'geometry',
    'guides',
    'physics',
    'control',
    'viz',
}


def test_public_surface():
    assert set(jxm.__all__) == EXPECTED
    for name in EXPECTED:
        assert hasattr(jxm, name)
    assert hasattr(jxm.physics, 'Morse')
    assert hasattr(jxm.physics, 'SaturatingCellGrowth')
    assert hasattr(jxm.physics, 'Division')
    assert hasattr(jxm.control, 'GeneNetworkConnectionist')
    assert hasattr(jxm.control, 'NeuralODE')
    assert hasattr(jxm.viz, 'draw')
    assert hasattr(jxm.viz, 'animate')
    assert hasattr(jxm.viz, 'plot_timeseries')
