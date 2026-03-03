def test_public_imports_smoke():
    import amphi_rl_dpgraph as pkg
    from amphi_rl_dpgraph import ContextState, ExposurePolicyController, apply_masking

    assert pkg is not None
    assert ContextState is not None
    assert ExposurePolicyController is not None
    assert apply_masking is not None
