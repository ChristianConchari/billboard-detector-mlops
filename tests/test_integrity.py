from vallavision.integrity import git_commit, sha256_dir


def test_sha256_dir_ignora_metadata_y_detecta_cambios(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"1")
    (tmp_path / "sub" / "b.bin").write_bytes(b"2")
    h = sha256_dir(tmp_path)
    (tmp_path / "metadata.json").write_text("{}")
    assert sha256_dir(tmp_path) == h
    (tmp_path / "sub" / "b.bin").write_bytes(b"3")
    assert sha256_dir(tmp_path) != h


def test_git_commit(tmp_path):
    (tmp_path / "refs" / "heads").mkdir(parents=True)
    (tmp_path / "HEAD").write_text("ref: refs/heads/main\n")
    assert git_commit(tmp_path) is None
    (tmp_path / "packed-refs").write_text("# pack\nabc123 refs/heads/main\n")
    assert git_commit(tmp_path) == "abc123"
    (tmp_path / "refs" / "heads" / "main").write_text("def456\n")
    assert git_commit(tmp_path) == "def456"
    assert git_commit(tmp_path / "no-existe") is None
