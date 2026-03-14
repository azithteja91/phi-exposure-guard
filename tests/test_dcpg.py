import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_node_id_format():
    from amphi_rl_dpgraph.dcpg import DCPGNode
    n = DCPGNode(patient_key="p1", phi_type="NAME", modality="text")
    assert n.node_id == "p1::text::NAME"

def test_edge_weight_all_ones():
    from amphi_rl_dpgraph.dcpg import DCPGEdge
    w = DCPGEdge.compute_weight(f_temporal=1.0, f_semantic=1.0, f_modality=1.0, f_trust=1.0)
    assert abs(w - 1.0) < 1e-9

def test_edge_weight_temporal_only():
    from amphi_rl_dpgraph.dcpg import DCPGEdge
    w = DCPGEdge.compute_weight(f_temporal=1.0, f_semantic=0.0, f_modality=0.0, f_trust=0.0)
    assert abs(w - 0.30) < 1e-9

def test_ngram_vector_length():
    from amphi_rl_dpgraph.dcpg import _ngram_vector
    assert len(_ngram_vector("hello world", dim=64)) == 64

def test_ngram_vector_sums_to_one():
    from amphi_rl_dpgraph.dcpg import _ngram_vector
    v = _ngram_vector("test text", dim=64)
    assert abs(sum(v) - 1.0) < 1e-6

def test_ngram_vector_empty():
    from amphi_rl_dpgraph.dcpg import _ngram_vector
    v = _ngram_vector("", dim=64)
    assert len(v) == 64 and sum(v) == 0.0

def test_cosine_similarity_identical():
    from amphi_rl_dpgraph.dcpg import _cosine_similarity
    v = [1.0, 0.0, 0.0, 1.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

def test_cosine_similarity_orthogonal():
    from amphi_rl_dpgraph.dcpg import _cosine_similarity
    assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6

def test_cosine_similarity_zero_vector():
    from amphi_rl_dpgraph.dcpg import _cosine_similarity
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

def test_is_cross_modal_text_asr():
    from amphi_rl_dpgraph.dcpg import _is_cross_modal
    assert not _is_cross_modal("text", "asr")

def test_is_cross_modal_text_image():
    from amphi_rl_dpgraph.dcpg import _is_cross_modal
    assert _is_cross_modal("text", "image_proxy")

def test_modality_to_phi_type():
    from amphi_rl_dpgraph.dcpg import _modality_to_phi_type
    assert _modality_to_phi_type("text") == "NAME_DATE_MRN_FACILITY"
    assert _modality_to_phi_type("audio_proxy") == "VOICE"
    assert _modality_to_phi_type("image_proxy") == "FACE_IMAGE"
