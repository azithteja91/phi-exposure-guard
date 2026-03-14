import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _tmp_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = f.name
    f.close()
    os.unlink(path)
    return path


def test_connect_creates_db():
    from amphi_rl_dpgraph.db import DBConfig, connect_db
    db_path = _tmp_db()
    try:
        conn = connect_db(DBConfig(db_path=db_path))
        assert conn is not None
        conn.close()
        assert Path(db_path).exists()
    finally:
        if Path(db_path).exists():
            os.unlink(db_path)


def test_open_context():
    from amphi_rl_dpgraph.db import open_context
    db_path = _tmp_db()
    try:
        conn = open_context(db_path)
        assert conn is not None
        conn.close()
    finally:
        if Path(db_path).exists():
            os.unlink(db_path)


def test_get_cross_modal_remask_count_no_table():
    from amphi_rl_dpgraph.db import open_context, get_cross_modal_remask_count
    db_path = _tmp_db()
    try:
        conn = open_context(db_path)
        assert get_cross_modal_remask_count(conn) == 0
        conn.close()
    finally:
        if Path(db_path).exists():
            os.unlink(db_path)


def test_wal_mode_set():
    from amphi_rl_dpgraph.db import DBConfig, connect_db
    db_path = _tmp_db()
    try:
        conn = connect_db(DBConfig(db_path=db_path))
        row = conn.execute("PRAGMA journal_mode;").fetchone()
        assert row[0].lower() == "wal"
        conn.close()
    finally:
        if Path(db_path).exists():
            os.unlink(db_path)
