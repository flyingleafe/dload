import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from dload.cli import cli
from dload.pack import PackReader
from dload.remote import LocalRemote
from dload.repo import Repository

# -- fixtures ------------------------------------------------------------------


@pytest.fixture
def bare_env(tmp_path, monkeypatch):
    """cwd + HOME/XDG isolated, no remote configured."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdgconf"))
    monkeypatch.setenv("DLOAD_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("DLOAD_BUCKET", raising=False)
    monkeypatch.delenv("DLOAD_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("DLOAD_PREFIX", raising=False)
    monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    return tmp_path


@pytest.fixture
def repo_env(bare_env, monkeypatch):
    """A working Repository backed by LocalRemote, no network involved."""
    remote_root = bare_env / "remote"
    monkeypatch.setattr("dload.repo.S3Remote", lambda *a, **k: LocalRemote(remote_root))
    monkeypatch.setenv("DLOAD_ENDPOINT_URL", "http://local.example")
    monkeypatch.setenv("DLOAD_BUCKET", "test-bucket")
    return bare_env


def _commit(repo_env: Path, name: str, files: dict[str, bytes]) -> str:
    data_dir = repo_env / f"data-{name}"
    for rel, content in files.items():
        p = data_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    result = CliRunner().invoke(cli, ["commit", name, "--from", str(data_dir)])
    assert result.exit_code == 0, result.output
    return result.output.strip().splitlines()[-1]


def _read_shards(repo: Repository, manifest) -> dict:
    samples = {}
    for shard_info in manifest.shards:
        with repo.open_shard(shard_info) as path, PackReader(path) as reader:
            for key, fields in reader:
                samples[key] = fields
    return samples


# -- --help ----------------------------------------------------------------


def test_help_works():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Commands:" in result.output


# -- init --------------------------------------------------------------------


def test_init_project_writes_file_and_prints_config(bare_env):
    result = CliRunner().invoke(
        cli,
        [
            "init",
            "--project",
            "--endpoint-url",
            "https://ep.example.com",
            "--bucket",
            "proj-bucket",
            "--cache-budget",
            "10GB",
        ],
    )
    assert result.exit_code == 0, result.output
    toml_path = Path.cwd() / "dload.toml"
    assert toml_path.exists()
    text = toml_path.read_text()
    assert "proj-bucket" in text
    assert "ep.example.com" in text
    assert "wrote" in result.output
    assert "proj-bucket" in result.output


def test_init_user_writes_xdg_config(bare_env):
    result = CliRunner().invoke(cli, ["init", "--bucket", "user-bucket"])
    assert result.exit_code == 0, result.output
    xdg_path = bare_env / "xdgconf" / "dload" / "config.toml"
    assert xdg_path.exists()
    assert "user-bucket" in xdg_path.read_text()
    assert not (Path.cwd() / "dload.toml").exists()


def test_init_only_updates_provided_options(bare_env):
    CliRunner().invoke(
        cli,
        [
            "init",
            "--project",
            "--endpoint-url",
            "https://a.example.com",
            "--bucket",
            "first-bucket",
            "--cache-dir",
            "/scratch/x",
        ],
    )
    result = CliRunner().invoke(cli, ["init", "--project", "--bucket", "second-bucket"])
    assert result.exit_code == 0, result.output
    text = (Path.cwd() / "dload.toml").read_text()
    assert "second-bucket" in text
    assert "https://a.example.com" in text
    assert "/scratch/x" in text


# -- status --------------------------------------------------------------------


def test_status_reports_remote_ok(repo_env):
    result = CliRunner().invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "remote: ok" in result.output
    assert "cache:" in result.output


def test_status_reports_remote_error_when_unconfigured(bare_env):
    result = CliRunner().invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "remote: error" in result.output


# -- commit --------------------------------------------------------------------


def test_commit_groups_files_into_samples(repo_env):
    data_dir = repo_env / "src_data"
    (data_dir / "a").mkdir(parents=True)
    (data_dir / "b").mkdir()
    (data_dir / "a" / "x.wav").write_bytes(b"WAVDATA")
    (data_dir / "a" / "x.json").write_bytes(b'{"k": 1}')
    (data_dir / "b" / "y.txt").write_bytes(b"hello")
    (data_dir / "plain").write_bytes(b"raw-bytes")

    result = CliRunner().invoke(cli, ["commit", "myds", "--from", str(data_dir)])
    assert result.exit_code == 0, result.output
    version = result.output.strip().splitlines()[-1]
    assert len(version) == 64

    repo = Repository.open()
    manifest = repo.manifest("myds")
    assert manifest.version == version
    assert manifest.num_samples == 3

    samples = _read_shards(repo, manifest)
    assert samples == {
        "a/x": {"wav": b"WAVDATA", "json": b'{"k": 1}'},
        "b/y": {"txt": b"hello"},
        "plain": {"data": b"raw-bytes"},
    }


def test_commit_with_meta_and_recipe(repo_env):
    data_dir = repo_env / "src2"
    data_dir.mkdir()
    (data_dir / "sample").write_bytes(b"content")
    recipe_file = repo_env / "recipe.sh"
    recipe_file.write_text("#!/bin/sh\necho hi\n")

    result = CliRunner().invoke(
        cli,
        [
            "commit",
            "recipeds",
            "--from",
            str(data_dir),
            "--recipe",
            str(recipe_file),
            "--meta",
            "source=unit-test",
            "--meta",
            "lang=en",
        ],
    )
    assert result.exit_code == 0, result.output

    repo = Repository.open()
    manifest = repo.manifest("recipeds")
    assert manifest.meta == {"source": "unit-test", "lang": "en"}
    assert manifest.recipe == "#!/bin/sh\necho hi\n"


def test_commit_bad_meta_is_rejected(repo_env):
    data_dir = repo_env / "src3"
    data_dir.mkdir()
    (data_dir / "sample").write_bytes(b"x")
    result = CliRunner().invoke(
        cli, ["commit", "badmeta", "--from", str(data_dir), "--meta", "no-equals-sign"]
    )
    assert result.exit_code != 0


# -- ls --------------------------------------------------------------------


def test_ls_empty_store(repo_env):
    result = CliRunner().invoke(cli, ["ls"])
    assert result.exit_code == 0, result.output
    assert "no datasets" in result.output


def test_ls_lists_datasets(repo_env):
    _commit(repo_env, "dsA", {"one": b"12345"})
    result = CliRunner().invoke(cli, ["ls"])
    assert result.exit_code == 0, result.output
    assert "dsA" in result.output
    assert "NAME" in result.output


# -- info --------------------------------------------------------------------


def test_info_shows_manifest_and_versions(repo_env):
    v1 = _commit(repo_env, "dsB", {"f": b"aaa"})
    v2 = _commit(repo_env, "dsB", {"f": b"bbb"})

    result = CliRunner().invoke(cli, ["info", "dsB"])
    assert result.exit_code == 0, result.output
    assert "name: dsB" in result.output
    assert f"version: {v2}" in result.output
    assert v1[:12] in result.output
    assert v2[:12] in result.output
    assert "latest" in result.output

    CliRunner().invoke(cli, ["pin", "dsB", v1])
    result2 = CliRunner().invoke(cli, ["info", "dsB", "--version", v2[:12]])
    assert result2.exit_code == 0, result2.output
    assert f"version: {v2}" in result2.output
    assert "pinned" in result2.output


# -- recipe --------------------------------------------------------------------


def test_recipe_prints_verbatim(repo_env):
    data_dir = repo_env / "data-rec"
    data_dir.mkdir()
    (data_dir / "f").write_bytes(b"x")
    recipe_file = repo_env / "recipe.txt"
    recipe_file.write_text("line one\nline two\n")
    CliRunner().invoke(
        cli,
        ["commit", "recipeds2", "--from", str(data_dir), "--recipe", str(recipe_file)],
    )

    result = CliRunner().invoke(cli, ["recipe", "recipeds2"])
    assert result.exit_code == 0, result.output
    assert result.output == "line one\nline two\n"


def test_recipe_missing_errors(repo_env):
    _commit(repo_env, "norecipe", {"f": b"x"})
    result = CliRunner().invoke(cli, ["recipe", "norecipe"])
    assert result.exit_code != 0
    assert "no stored recipe" in result.output


# -- pin / unpin -----------------------------------------------------------------


def test_pin_and_unpin_manage_lock_file(repo_env):
    v1 = _commit(repo_env, "dsC", {"f": b"1"})
    v2 = _commit(repo_env, "dsC", {"f": b"2"})

    result = CliRunner().invoke(cli, ["pin", "dsC", v1[:12]])
    assert result.exit_code == 0, result.output
    lock_path = Path.cwd() / "dload.lock"
    assert lock_path.exists()
    locked = tomllib.loads(lock_path.read_text())
    assert locked["datasets"]["dsC"] == v1

    repo = Repository.open()
    assert repo.resolve("dsC") == v1

    result2 = CliRunner().invoke(cli, ["unpin", "dsC"])
    assert result2.exit_code == 0, result2.output
    assert "dsC" not in tomllib.loads(lock_path.read_text()).get("datasets", {})
    assert repo.resolve("dsC") == v2


def test_pin_without_version_uses_latest(repo_env):
    _commit(repo_env, "dsD", {"f": b"1"})
    v2 = _commit(repo_env, "dsD", {"f": b"2"})
    result = CliRunner().invoke(cli, ["pin", "dsD"])
    assert result.exit_code == 0, result.output
    assert v2 in result.output


# -- pull --------------------------------------------------------------------


def test_pull_fetches_shards_into_cache(repo_env):
    _commit(repo_env, "dsE", {"f": b"payload-bytes"})
    result = CliRunner().invoke(cli, ["pull", "dsE"])
    assert result.exit_code == 0, result.output
    assert "pulled" in result.output

    cache_dir = repo_env / "cache"
    shard_files = [p for p in (cache_dir / "shards").rglob("*") if p.is_file()]
    assert len(shard_files) == 1


# -- cache status / clear -----------------------------------------------------


def test_cache_status_and_clear(repo_env):
    _commit(repo_env, "dsF", {"f": b"data-for-cache"})
    CliRunner().invoke(cli, ["pull", "dsF"])

    status_result = CliRunner().invoke(cli, ["cache", "status"])
    assert status_result.exit_code == 0, status_result.output
    assert "shards: 1" in status_result.output

    clear_result = CliRunner().invoke(cli, ["cache", "clear", "--yes"])
    assert clear_result.exit_code == 0, clear_result.output
    assert "freed" in clear_result.output

    status_result2 = CliRunner().invoke(cli, ["cache", "status"])
    assert "shards: 0" in status_result2.output


# -- rm + gc --------------------------------------------------------------------


def test_rm_without_yes_prompts_and_aborts_on_no(repo_env):
    _commit(repo_env, "dsI", {"f": b"content"})
    result = CliRunner().invoke(cli, ["rm", "dsI"], input="n\n")
    assert result.exit_code != 0
    ls_result = CliRunner().invoke(cli, ["ls"])
    assert "dsI" in ls_result.output


def test_rm_specific_version_keeps_other_versions(repo_env):
    v1 = _commit(repo_env, "dsJ", {"f": b"v1content"})
    v2 = _commit(repo_env, "dsJ", {"f": b"v2content"})

    result = CliRunner().invoke(cli, ["rm", "dsJ", "--version", v1[:12], "--yes"])
    assert result.exit_code == 0, result.output

    repo = Repository.open()
    versions = repo.versions("dsJ")
    assert len(versions) == 1
    assert versions[0].version == v2


def test_rm_and_gc_flow(repo_env):
    _commit(repo_env, "dsG", {"f": b"unique-content-1"})
    _commit(repo_env, "dsH", {"f": b"unique-content-2"})

    rm_result = CliRunner().invoke(cli, ["rm", "dsG", "--yes"])
    assert rm_result.exit_code == 0, rm_result.output

    ls_result = CliRunner().invoke(cli, ["ls"])
    assert "dsG" not in ls_result.output
    assert "dsH" in ls_result.output

    remote_root = repo_env / "remote"
    shards_before = [p for p in (remote_root / "shards").rglob("*") if p.is_file()]

    gc_result = CliRunner().invoke(cli, ["gc", "--yes"])
    assert gc_result.exit_code == 0, gc_result.output
    assert "freed" in gc_result.output

    shards_after = [p for p in (remote_root / "shards").rglob("*") if p.is_file()]
    assert len(shards_after) < len(shards_before)
