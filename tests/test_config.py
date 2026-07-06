import os
from pathlib import Path

import pytest

from dload.config import Config, format_size, parse_size, write_config_file
from dload.errors import ConfigError

# -- parse_size / format_size -------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", 0),
        ("100", 100),
        ("50GB", 50_000_000_000),
        ("1.5TiB", int(1.5 * 1024**4)),
        ("800M", 800_000_000),
        ("800MB", 800_000_000),
        ("1KiB", 1024),
        ("2GiB", 2 * 1024**3),
        ("  50 GB  ", 50_000_000_000),
        ("50gb", 50_000_000_000),
    ],
)
def test_parse_size_units(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["unlimited", "UNLIMITED", "  Unlimited "])
def test_parse_size_unlimited(text):
    assert parse_size(text) is None


@pytest.mark.parametrize("text", ["", "garbage", "GB", "-5GB", "50XB", "50 GB extra"])
def test_parse_size_garbage_raises(text):
    with pytest.raises(ConfigError):
        parse_size(text)


def test_format_size_unlimited():
    assert format_size(None) == "unlimited"


def test_format_size_bytes():
    assert format_size(500) == "500 B"


def test_format_size_binary_units():
    assert format_size(1024) == "1.0 KiB"
    assert format_size(1024**3) == "1.0 GiB"


def test_format_size_roundish():
    assert format_size(int(1.5 * 1024**3)) == "1.5 GiB"


# -- layered config resolution -------------------------------------------------


def test_defaults_when_nothing_set(tmp_path):
    env = {"HOME": str(tmp_path)}
    cfg = Config.load(cwd=tmp_path, env=env)
    assert cfg.endpoint_url is None
    assert cfg.bucket is None
    assert cfg.prefix == ""
    assert cfg.cache_dir == tmp_path / ".cache" / "dload"
    assert cfg.cache_budget_raw == "auto"
    assert isinstance(cfg.cache_budget, int)


def test_load_defaults_to_os_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("DLOAD_BUCKET", "real-environ-bucket")
    monkeypatch.delenv("DLOAD_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)
    cfg = Config.load(cwd=tmp_path)
    assert cfg.bucket == "real-environ-bucket"


def test_env_vars_win_over_everything(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "dload.toml").write_text(
        '[remote]\nendpoint_url = "https://file.example.com"\nbucket = "file-bucket"\n'
    )
    user_config = tmp_path / "home" / ".config" / "dload" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text('[remote]\nbucket = "user-bucket"\n')

    env = {
        "HOME": str(tmp_path / "home"),
        "DLOAD_ENDPOINT_URL": "https://env.example.com",
        "DLOAD_BUCKET": "env-bucket",
    }
    cfg = Config.load(cwd=project_dir, env=env)
    assert cfg.endpoint_url == "https://env.example.com"
    assert cfg.bucket == "env-bucket"


def test_project_file_wins_over_user_file(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "dload.toml").write_text('[remote]\nbucket = "project-bucket"\n')
    user_config = tmp_path / "home" / ".config" / "dload" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text('[remote]\nbucket = "user-bucket"\n')

    env = {"HOME": str(tmp_path / "home")}
    cfg = Config.load(cwd=project_dir, env=env)
    assert cfg.bucket == "project-bucket"


def test_project_file_found_from_nested_ancestor(tmp_path):
    project_dir = tmp_path / "project"
    nested = project_dir / "sub" / "deeper"
    nested.mkdir(parents=True)
    (project_dir / "dload.toml").write_text('[remote]\nbucket = "ancestor-bucket"\n')

    env = {"HOME": str(tmp_path / "home")}
    cfg = Config.load(cwd=nested, env=env)
    assert cfg.bucket == "ancestor-bucket"


def test_user_file_used_when_no_project_file(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    user_config = tmp_path / "home" / ".config" / "dload" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(
        '[remote]\nbucket = "user-bucket"\n[cache]\nbudget = "unlimited"\n'
    )

    env = {"HOME": str(tmp_path / "home")}
    cfg = Config.load(cwd=project_dir, env=env)
    assert cfg.bucket == "user-bucket"
    assert cfg.cache_budget is None
    assert cfg.cache_budget_raw == "unlimited"


def test_xdg_config_home_respected(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    xdg_config = tmp_path / "xdgconf"
    user_config = xdg_config / "dload" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text('[remote]\nbucket = "xdg-bucket"\n')

    env = {"HOME": str(tmp_path / "home"), "XDG_CONFIG_HOME": str(xdg_config)}
    cfg = Config.load(cwd=project_dir, env=env)
    assert cfg.bucket == "xdg-bucket"


def test_xdg_cache_home_respected(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    xdg_cache = tmp_path / "xdgcache"

    env = {"HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(xdg_cache)}
    cfg = Config.load(cwd=project_dir, env=env)
    assert cfg.cache_dir == xdg_cache / "dload"


def test_explicit_cache_dir_and_budget_from_env(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    cache_dir = tmp_path / "scratch"
    cache_dir.mkdir()

    env = {
        "HOME": str(tmp_path / "home"),
        "DLOAD_CACHE_DIR": str(cache_dir),
        "DLOAD_CACHE_BUDGET": "50GB",
    }
    cfg = Config.load(cwd=project_dir, env=env)
    assert cfg.cache_dir == cache_dir
    assert cfg.cache_budget == 50_000_000_000
    assert cfg.cache_budget_raw == "50GB"


def test_auto_budget_is_half_free_space(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    cache_dir = tmp_path / "cache"

    env = {"HOME": str(tmp_path / "home"), "DLOAD_CACHE_DIR": str(cache_dir)}
    cfg = Config.load(cwd=project_dir, env=env)

    st = os.statvfs(tmp_path)  # nearest existing ancestor of cache_dir
    expected = (st.f_bavail * st.f_frsize) // 2
    assert cfg.cache_budget == expected


# -- R2_ACCOUNT_ID derivation --------------------------------------------------


def test_r2_account_id_derives_endpoint(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    env = {"HOME": str(tmp_path / "home"), "R2_ACCOUNT_ID": "abc123"}
    cfg = Config.load(cwd=project_dir, env=env)
    assert cfg.endpoint_url == "https://abc123.r2.cloudflarestorage.com"


def test_explicit_endpoint_url_wins_over_r2_derivation(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    env = {
        "HOME": str(tmp_path / "home"),
        "R2_ACCOUNT_ID": "abc123",
        "DLOAD_ENDPOINT_URL": "https://custom.example.com",
    }
    cfg = Config.load(cwd=project_dir, env=env)
    assert cfg.endpoint_url == "https://custom.example.com"


def test_r2_derived_endpoint_wins_over_file(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "dload.toml").write_text(
        '[remote]\nendpoint_url = "https://file.example.com"\n'
    )
    env = {"HOME": str(tmp_path / "home"), "R2_ACCOUNT_ID": "abc123"}
    cfg = Config.load(cwd=project_dir, env=env)
    assert cfg.endpoint_url == "https://abc123.r2.cloudflarestorage.com"


# -- require_remote -------------------------------------------------------------


def test_require_remote_returns_pair_when_set(tmp_path):
    env = {
        "HOME": str(tmp_path / "home"),
        "DLOAD_ENDPOINT_URL": "https://env.example.com",
        "DLOAD_BUCKET": "env-bucket",
    }
    cfg = Config.load(cwd=tmp_path, env=env)
    assert cfg.require_remote() == ("https://env.example.com", "env-bucket")


def test_require_remote_raises_with_helpful_message(tmp_path):
    env = {"HOME": str(tmp_path / "home")}
    cfg = Config.load(cwd=tmp_path, env=env)
    with pytest.raises(ConfigError) as exc_info:
        cfg.require_remote()
    message = str(exc_info.value)
    assert "endpoint_url" in message
    assert "bucket" in message
    assert "DLOAD_ENDPOINT_URL" in message or "DLOAD_BUCKET" in message


def test_require_remote_partial_missing_mentions_only_missing_field(tmp_path):
    env = {
        "HOME": str(tmp_path / "home"),
        "DLOAD_ENDPOINT_URL": "https://env.example.com",
    }
    cfg = Config.load(cwd=tmp_path, env=env)
    with pytest.raises(ConfigError) as exc_info:
        cfg.require_remote()
    message = str(exc_info.value)
    assert "bucket" in message


# -- write_config_file ----------------------------------------------------------


def test_write_config_file_creates_new(tmp_path):
    path = tmp_path / "dload.toml"
    write_config_file(path, endpoint_url="https://x.example.com", bucket="my-bucket")
    assert path.exists()

    cfg = Config.load(cwd=tmp_path, env={"HOME": str(tmp_path / "home")})
    assert cfg.endpoint_url == "https://x.example.com"
    assert cfg.bucket == "my-bucket"


def test_write_config_file_update_preserves_other_keys(tmp_path):
    path = tmp_path / "dload.toml"
    write_config_file(
        path,
        endpoint_url="https://x.example.com",
        bucket="my-bucket",
        cache_dir="/scratch/cache",
    )
    write_config_file(path, bucket="new-bucket")

    cfg = Config.load(cwd=tmp_path, env={"HOME": str(tmp_path / "home")})
    assert cfg.endpoint_url == "https://x.example.com"
    assert cfg.bucket == "new-bucket"
    assert cfg.cache_dir == Path("/scratch/cache")


def test_write_config_file_handles_special_characters(tmp_path):
    path = tmp_path / "dload.toml"
    write_config_file(path, prefix='has "quotes" and \\backslash')
    cfg = Config.load(cwd=tmp_path, env={"HOME": str(tmp_path / "home")})
    assert cfg.prefix == 'has "quotes" and \\backslash'
