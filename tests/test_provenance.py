from rfm_structure.provenance import (
    file_identity,
    preprocessing_identity,
    software_versions,
)


def test_file_identity_is_a_known_streaming_sha256(tmp_path):
    path = tmp_path / "fixture"
    path.write_bytes(b"abc")
    identity = file_identity(path)
    assert identity["bytes"] == 3
    assert identity["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_preprocessing_identity_includes_embeddings_and_relation_metadata(tmp_path):
    directory = tmp_path / "rel-f1"
    directory.mkdir()
    names = ["meta.json", "table_info.json", "column_index.json", "relation_index.json", "nodes.rkyv",
             "offsets.rkyv", "p2f_adj.rkyv", "text_emb_all-MiniLM-L12-v2.bin"]
    for name in names:
        (directory / name).write_bytes(name.encode())
    identity = preprocessing_identity(tmp_path, "rel-f1")
    assert set(identity) == set(names)
    assert identity["relation_index.json"]["sha256"] != identity["nodes.rkyv"]["sha256"]
    assert software_versions()["numpy"] is not None
