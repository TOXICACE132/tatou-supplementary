## server/test/test_watermark_endpoints_branch.py
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from server.src.watermarking_method import (
    WatermarkingMethod,
    SecretNotFoundError,
    InvalidKeyError,
    load_pdf_bytes,
)

# ============================================================
# 1) Custom mock watermarking methods (assignment requirement)
# ============================================================
# These methods are "custom watermarking methods" required by the assignment.
# They allow deterministic success/failure behaviors without depending on real
# watermark algorithms, so we can drive server branches reliably.

class FakeOK(WatermarkingMethod):
    """A minimal method that always succeeds and embeds data at the end."""
    name = "fake-ok"

    @staticmethod
    def get_usage() -> str:
        return "fake-ok: always succeeds"

    def is_watermark_applicable(self, pdf, position=None) -> bool:
        # Ensures input is a readable PDF payload (or raises).
        load_pdf_bytes(pdf)
        return True

    def add_watermark(self, pdf, secret: str, key: str, position=None) -> bytes:
        # Appends a deterministic marker to the PDF bytes.
        b = load_pdf_bytes(pdf)
        return b + f"\n%FAKE:{secret}:{key}\n".encode()

    def read_secret(self, pdf, key: str) -> str:
        # Reads back our fake marker and validates the embedded key.
        b = load_pdf_bytes(pdf)
        marker = b"%FAKE:"
        if marker not in b:
            raise SecretNotFoundError("no fake watermark")
        tail = b.split(marker, 1)[1].split(b"\n", 1)[0]  # secret:key
        secret, embedded_key = tail.split(b":", 1)
        if embedded_key.decode() != key:
            raise InvalidKeyError("bad key")
        return secret.decode()


class FakeEmptyOut(WatermarkingMethod):
    """Returns empty bytes -> should trigger 'produced no output' branch."""
    name = "fake-empty"

    @staticmethod
    def get_usage() -> str:
        return "fake-empty: returns empty bytes"

    def is_watermark_applicable(self, pdf, position=None) -> bool:
        load_pdf_bytes(pdf)
        return True

    def add_watermark(self, pdf, secret: str, key: str, position=None) -> bytes:
        # Explicitly returns b"" to trigger server-side "produced no output" branch.
        load_pdf_bytes(pdf)
        return b""

    def read_secret(self, pdf, key: str) -> str:
        raise SecretNotFoundError("always missing")


class FakeBoom(WatermarkingMethod):
    """Raises exceptions -> used to cover exception branches."""
    name = "fake-boom"

    @staticmethod
    def get_usage() -> str:
        return "fake-boom: raises exception"

    def is_watermark_applicable(self, pdf, position=None) -> bool:
        # Force applicability check failure branch.
        load_pdf_bytes(pdf)
        raise RuntimeError("applicability boom")

    def add_watermark(self, pdf, secret: str, key: str, position=None) -> bytes:
        # Force watermarking apply failure branch.
        raise RuntimeError("add boom")

    def read_secret(self, pdf, key: str) -> str:
        # Force read failure branch.
        raise RuntimeError("read boom")


@pytest.fixture
def fake_methods_registered():
    """
    Register our custom methods into WMUtils.METHODS for this test module,
    then restore the registry after tests.

    Why:
    - Assignment requires custom mock watermark methods.
    - Also allows end-to-end flow without relying on real watermarking code.
    """
    import server.src.server as server_module

    old = dict(server_module.WMUtils.METHODS)

    server_module.WMUtils.METHODS["fake-ok"] = FakeOK()
    server_module.WMUtils.METHODS["fake-empty"] = FakeEmptyOut()
    server_module.WMUtils.METHODS["fake-boom"] = FakeBoom()

    yield

    # Restore original registry to avoid cross-test pollution.
    server_module.WMUtils.METHODS.clear()
    server_module.WMUtils.METHODS.update(old)


# ============================================================
# 2) Helpers for patching DB engine/connection
# ============================================================
# These helpers let us simulate SQLAlchemy engine/connect/begin context managers
# used inside server.py, without touching a real database.

def _mk_select_row(doc_id=1, name="doc.pdf", path="files/u/doc.pdf"):
    """Build a fake row returned from SELECT Documents."""
    row = MagicMock()
    row.id = doc_id
    row.name = name
    row.path = path
    return row


def _mk_engine_for_select(row_or_exc):
    """
    Creates an engine where:
      with engine.connect() as conn:
          conn.execute(...).first() returns row_or_exc (or raises)

    Used to simulate:
    - Document SELECT success (return a row)
    - Document not found (return None)
    - DB failure (raise exception)
    """
    engine = MagicMock()
    conn = MagicMock()

    if isinstance(row_or_exc, Exception):
        conn.execute.side_effect = row_or_exc
    else:
        res = MagicMock()
        res.first.return_value = row_or_exc
        conn.execute.return_value = res

    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    engine.connect.return_value = cm
    return engine, conn


def _mk_engine_for_insert(insert_side_effect=None, select_existing_row=None, vid_lastrowid=77):
    """
    Creates an engine where:
      with engine.begin() as conn:   # INSERT
          conn.execute(...) -> lastrowid or raises insert_side_effect

      with engine.connect() as conn: # SELECT existing version id (duplicate branch)
          conn.execute(...).first() -> select_existing_row

    Used to simulate:
    - Normal version insert (returns lastrowid)
    - IntegrityError insert
    - Generic exception insert
    - Duplicate-link recovery (select existing row id)
    """
    engine = MagicMock()

    # begin() -> conn_insert
    conn_insert = MagicMock()
    if insert_side_effect is None:
        insert_res = MagicMock()
        insert_res.lastrowid = vid_lastrowid
        conn_insert.execute.return_value = insert_res
    else:
        conn_insert.execute.side_effect = insert_side_effect

    begin_cm = MagicMock()
    begin_cm.__enter__.return_value = conn_insert
    begin_cm.__exit__.return_value = False
    engine.begin.return_value = begin_cm

    # connect() -> conn_select (used in duplicate branch)
    conn_select = MagicMock()
    sel_res = MagicMock()
    sel_res.first.return_value = select_existing_row
    conn_select.execute.return_value = sel_res

    conn_cm = MagicMock()
    conn_cm.__enter__.return_value = conn_select
    conn_cm.__exit__.return_value = False
    engine.connect.return_value = conn_cm

    return engine


def _mk_engine_connect_sequence(begin_engine, connect_cms_in_order):
    """
    Build a "hybrid engine" whose:
      - begin() comes from begin_engine.begin
      - connect() returns different context managers per call (sequence)

    Why this exists:
    - create_watermark calls get_engine(app).connect() multiple times.
      The first connect() is for Documents SELECT.
      The second connect() (only in duplicate-link branch) is for Versions SELECT.
    - We need connect() to behave differently depending on call order.
    """
    hybrid = MagicMock()
    hybrid.begin = begin_engine.begin
    hybrid.connect = MagicMock(side_effect=connect_cms_in_order)
    return hybrid


def _write_min_pdf(fp: Path):
    """Write a tiny but valid PDF."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")


def _patch_get_method_returns_string_name(mocker, name_str: str):
    """
    server.py does: method_official = WMUtils.get_method(method).name

    So WMUtils.get_method must return an object with a *string* .name.
    If .name is a MagicMock, jsonify may fail or comparisons become unstable.
    """
    m = MagicMock()
    m.name = name_str
    mocker.patch("server.src.server.WMUtils.get_method", return_value=m)


# ============================================================
# 3) Happy path (custom method)
# ============================================================

def test_create_and_read_watermark_roundtrip_fake_ok(
    client, auth_headers, sample_pdf_path, fake_methods_registered
):
    """
    Integration-style behavior test (happy path):

    1) Upload a clean PDF
    2) Create watermark using custom method fake-ok
    3) Download generated version (get-version)
    4) Upload the downloaded PDF as a new document
    5) Read watermark back using fake-ok and verify the secret matches

    This validates that the primary routes create-watermark + read-watermark
    work together end-to-end under deterministic watermark method.
    """
    r = client.post(
        "/api/upload-document",
        data={"file": (io.BytesIO(sample_pdf_path.read_bytes()), "clean.pdf")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert r.status_code == 201
    doc_id = r.get_json()["id"]

    r2 = client.post(
        f"/api/create-watermark/{doc_id}",
        headers=auth_headers,
        json={
            "method": "fake-ok",
            "intended_for": "test",
            "secret": "S",
            "key": "K",
            "position": "eof",
        },
    )
    assert r2.status_code == 201
    link = r2.get_json()["link"]

    r3 = client.get(f"/api/get-version/{link}")
    assert r3.status_code == 200
    wm_bytes = r3.data

    r4 = client.post(
        "/api/upload-document",
        data={"file": (io.BytesIO(wm_bytes), "wm.pdf")},
        headers=auth_headers,
        content_type="multipart/form-data",
    )
    assert r4.status_code == 201
    new_doc_id = r4.get_json()["id"]

    r5 = client.post(
        f"/api/read-watermark/{new_doc_id}",
        headers=auth_headers,
        json={"method": "fake-ok", "key": "K", "position": "eof"},
    )
    assert r5.status_code == 200
    assert r5.get_json()["secret"] == "S"


# ============================================================
# 4) create_watermark branch coverage
# ============================================================

def test_create_watermark_bad_document_id_returns_400(client, auth_headers):
    """
    Validates "document_id parsing" failure branch:

    - Calling /api/create-watermark/<int:document_id> with a non-int path segment
      may be rejected by Flask routing (404 or 405 depending on route config).
    - Then we force hitting server-side int() conversion by using query param id=abc,
      which must return 400 "document id required and must be integer".
    """
    r = client.post(
        "/api/create-watermark/not-an-int",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code in (404, 405)

    r2 = client.post(
        "/api/create-watermark?id=abc",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r2.status_code == 400


def test_create_watermark_missing_required_fields_returns_400(client, auth_headers):
    """
    Validates "missing JSON required fields" branch:

    Server requires: method, intended_for, secret(str), key(str).
    If missing -> return 400.
    """
    r = client.post("/api/create-watermark/1", headers=auth_headers, json={})
    assert r.status_code == 400


def test_create_watermark_db_error_on_select_returns_503(mocker, client, auth_headers):
    """
    Validates DB SELECT failure branch:

    If Documents SELECT raises any exception -> return 503.
    """
    eng, _ = _mk_engine_for_select(Exception("db down"))
    mocker.patch("server.src.server.get_engine", return_value=eng)

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 503


def test_create_watermark_document_not_found_returns_404(mocker, client, auth_headers):
    """
    Validates "document not found" branch:

    If Documents SELECT returns None -> return 404.
    """
    eng, _ = _mk_engine_for_select(None)
    mocker.patch("server.src.server.get_engine", return_value=eng)

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 404


def test_create_watermark_path_invalid_returns_500(mocker, client, auth_headers, tmp_path):
    """
    Validates "document path invalid (escapes storage dir)" branch:

    If row.path points outside STORAGE_DIR -> return 500.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    row = _mk_select_row(doc_id=1, name="doc.pdf", path="/etc/passwd")
    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 500


def test_create_watermark_file_missing_returns_410(mocker, client, auth_headers, tmp_path):
    """
    Validates "file missing on disk" branch:

    If resolved document path does not exist -> return 410.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    row = _mk_select_row(doc_id=1, name="doc.pdf", path="files/u/missing.pdf")
    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 410


def test_create_watermark_not_applicable_returns_400(mocker, client, auth_headers, tmp_path):
    """
    Validates "method not applicable" branch:

    If WMUtils.is_watermarking_applicable returns False -> return 400.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)
    mocker.patch("server.src.server.WMUtils.is_watermarking_applicable", return_value=False)

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 400


def test_create_watermark_applicability_exception_returns_400(mocker, client, auth_headers, tmp_path):
    """
    Validates "applicability check exception" branch:

    If WMUtils.is_watermarking_applicable raises -> return 400.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)
    mocker.patch("server.src.server.WMUtils.is_watermarking_applicable", side_effect=RuntimeError("boom"))

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 400


def test_create_watermark_apply_returns_empty_bytes_500(mocker, client, auth_headers, tmp_path):
    """
    Validates "watermarking produced no output" branch:

    If WMUtils.apply_watermark returns empty bytes -> return 500.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)
    mocker.patch("server.src.server.WMUtils.is_watermarking_applicable", return_value=True)
    mocker.patch("server.src.server.WMUtils.apply_watermark", return_value=b"")

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 500


def test_create_watermark_apply_raises_500(mocker, client, auth_headers, tmp_path):
    """
    Validates "watermarking failed (exception)" branch:

    If WMUtils.apply_watermark raises -> return 500.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)
    mocker.patch("server.src.server.WMUtils.is_watermarking_applicable", return_value=True)
    mocker.patch("server.src.server.WMUtils.apply_watermark", side_effect=RuntimeError("wm failed"))

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 500


def test_create_watermark_write_file_fails_500(mocker, client, auth_headers, tmp_path):
    """
    Validates "failed to write watermarked file" branch:

    We patch pathlib.Path.open to raise OSError when writing into watermarks/ dir.
    Server should respond 500.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)

    mocker.patch("server.src.server.WMUtils.is_watermarking_applicable", return_value=True)
    mocker.patch("server.src.server.WMUtils.apply_watermark", return_value=b"%PDF-1.4\n%%EOF\n")
    _patch_get_method_returns_string_name(mocker, "fake-ok")

    real_open = Path.open

    def bad_open(self, mode="r", *args, **kwargs):
        if "wb" in mode and str(self).endswith(".pdf") and "watermarks" in str(self):
            raise OSError("disk full")
        return real_open(self, mode, *args, **kwargs)

    mocker.patch("pathlib.Path.open", new=bad_open)

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 500


def test_create_watermark_versions_insert_duplicate_entry_returns_existing_201(mocker, client, auth_headers, tmp_path):
    """
    KEY TARGET (intended to cover server.py duplicate-link recovery branch):

    - Versions INSERT raises IntegrityError
    - The error message contains BOTH 'Duplicate entry' and 'uq_Versions_link'
    - Server then performs a SELECT on Versions table to locate the existing row
    - If a row is found, server returns 201 and uses existing row.id

    This requires connect() to return different context managers:
    - 1st connect() is for Documents SELECT (must return a document row)
    - 2nd connect() is for Versions SELECT (must return existing version row)
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    doc_row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    # Engine for Documents SELECT
    eng_doc, _ = _mk_engine_for_select(doc_row)

    # Engine for Versions INSERT + duplicate SELECT existing id
    db_exc = IntegrityError("Duplicate entry", None, Exception("Duplicate entry for uq_Versions_link"))

    existing = MagicMock()
    existing.id = 123

    eng_ver = _mk_engine_for_insert(insert_side_effect=db_exc, select_existing_row=existing)

    # IMPORTANT FIX:
    # connect() must return doc-select CM first, then version-select CM second.
    hybrid = _mk_engine_connect_sequence(
        begin_engine=eng_ver,
        connect_cms_in_order=[eng_doc.connect.return_value, eng_ver.connect.return_value],
    )
    mocker.patch("server.src.server.get_engine", return_value=hybrid)

    mocker.patch("server.src.server.WMUtils.is_watermarking_applicable", return_value=True)
    mocker.patch("server.src.server.WMUtils.apply_watermark", return_value=b"%PDF-1.4\n%%EOF\n")
    _patch_get_method_returns_string_name(mocker, "fake-ok")

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 201
    assert r.get_json()["id"] == 123


def test_create_watermark_versions_insert_integrityerror_other_returns_503(mocker, client, auth_headers, tmp_path):
    """
    Validates IntegrityError fallback branch:

    If IntegrityError does NOT match the duplicate-link keyword pattern,
    server should return 503.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    doc_row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    eng_doc, _ = _mk_engine_for_select(doc_row)
    db_exc = IntegrityError("Some integrity error", None, Exception("Other integrity"))
    eng_ver = _mk_engine_for_insert(insert_side_effect=db_exc, select_existing_row=None)

    hybrid = _mk_engine_connect_sequence(
        begin_engine=eng_ver,
        connect_cms_in_order=[eng_doc.connect.return_value, eng_ver.connect.return_value],
    )
    mocker.patch("server.src.server.get_engine", return_value=hybrid)

    mocker.patch("server.src.server.WMUtils.is_watermarking_applicable", return_value=True)
    mocker.patch("server.src.server.WMUtils.apply_watermark", return_value=b"%PDF-1.4\n%%EOF\n")
    _patch_get_method_returns_string_name(mocker, "fake-ok")

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 503


def test_create_watermark_versions_insert_integrityerror_other_unlink_raises_is_ignored(mocker, client, auth_headers, tmp_path):
    """
    Optional branch coverage for unlink() exception swallowing in IntegrityError branch:

    In the IntegrityError (non-duplicate) path, server attempts dest_path.unlink()
    and ignores exceptions. We force unlink() to raise to cover 'except: pass'.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    doc_row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    eng_doc, _ = _mk_engine_for_select(doc_row)
    db_exc = IntegrityError("Some integrity error", None, Exception("Other integrity"))
    eng_ver = _mk_engine_for_insert(insert_side_effect=db_exc, select_existing_row=None)

    hybrid = _mk_engine_connect_sequence(
        begin_engine=eng_ver,
        connect_cms_in_order=[eng_doc.connect.return_value, eng_ver.connect.return_value],
    )
    mocker.patch("server.src.server.get_engine", return_value=hybrid)

    mocker.patch("server.src.server.WMUtils.is_watermarking_applicable", return_value=True)
    mocker.patch("server.src.server.WMUtils.apply_watermark", return_value=b"%PDF-1.4\n%%EOF\n")
    _patch_get_method_returns_string_name(mocker, "fake-ok")

    # Force unlink() to raise; server should swallow it.
    mocker.patch("pathlib.Path.unlink", side_effect=OSError("cannot delete"))

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 503


def test_create_watermark_versions_insert_generic_exception_returns_503_unlink_raises_ignored(mocker, client, auth_headers, tmp_path):
    """
    Optional branch coverage for unlink() exception swallowing in generic Exception branch:

    In the generic Exception path during insert, server also tries dest_path.unlink()
    and ignores exceptions. We force unlink() to raise to cover 'except: pass'.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    doc_row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    eng_doc, _ = _mk_engine_for_select(doc_row)
    eng_ver = _mk_engine_for_insert(insert_side_effect=RuntimeError("insert boom"), select_existing_row=None)

    hybrid = _mk_engine_connect_sequence(
        begin_engine=eng_ver,
        connect_cms_in_order=[eng_doc.connect.return_value],
    )
    mocker.patch("server.src.server.get_engine", return_value=hybrid)

    mocker.patch("server.src.server.WMUtils.is_watermarking_applicable", return_value=True)
    mocker.patch("server.src.server.WMUtils.apply_watermark", return_value=b"%PDF-1.4\n%%EOF\n")
    _patch_get_method_returns_string_name(mocker, "fake-ok")

    mocker.patch("pathlib.Path.unlink", side_effect=OSError("cannot delete"))

    r = client.post(
        "/api/create-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "intended_for": "x", "secret": "S", "key": "K"},
    )
    assert r.status_code == 503


# ============================================================
# 5) read_watermark branch coverage
# ============================================================

def test_read_watermark_bad_document_id_400(client, auth_headers):
    """
    Validates document_id parsing failure branch in read-watermark:
    id=abc -> int() fails -> 400.
    """
    r = client.post(
        "/api/read-watermark?id=abc",
        headers=auth_headers,
        json={"method": "fake-ok", "key": "K"},
    )
    assert r.status_code == 400


def test_read_watermark_missing_fields_400(client, auth_headers):
    """
    Validates missing JSON fields branch in read-watermark:
    missing method/key -> 400.
    """
    r = client.post("/api/read-watermark/1", headers=auth_headers, json={})
    assert r.status_code == 400


def test_read_watermark_db_error_503(mocker, client, auth_headers):
    """
    Validates DB SELECT failure branch in read-watermark:
    Documents SELECT raises -> 503.
    """
    eng, _ = _mk_engine_for_select(Exception("db down"))
    mocker.patch("server.src.server.get_engine", return_value=eng)

    r = client.post(
        "/api/read-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "key": "K"},
    )
    assert r.status_code == 503


def test_read_watermark_document_not_found_404(mocker, client, auth_headers):
    """
    Validates not-found branch in read-watermark:
    Documents SELECT returns None -> 404.
    """
    eng, _ = _mk_engine_for_select(None)
    mocker.patch("server.src.server.get_engine", return_value=eng)

    r = client.post(
        "/api/read-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "key": "K"},
    )
    assert r.status_code == 404


def test_read_watermark_path_invalid_500(mocker, client, auth_headers, tmp_path):
    """
    Validates invalid path branch in read-watermark:
    path escapes STORAGE_DIR -> 500.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    row = _mk_select_row(doc_id=1, name="doc.pdf", path="/etc/passwd")
    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)

    r = client.post(
        "/api/read-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "key": "K"},
    )
    assert r.status_code == 500


def test_read_watermark_file_missing_410(mocker, client, auth_headers, tmp_path):
    """
    Validates file missing branch in read-watermark:
    resolved path does not exist -> 410.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    row = _mk_select_row(doc_id=1, name="doc.pdf", path="files/u/missing.pdf")
    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)

    r = client.post(
        "/api/read-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "key": "K"},
    )
    assert r.status_code == 410


def test_read_watermark_wmutils_raises_returns_400(mocker, client, auth_headers, tmp_path):
    """
    Validates WMUtils.read_watermark exception branch:
    if WMUtils.read_watermark raises -> 400.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)
    mocker.patch("server.src.server.WMUtils.read_watermark", side_effect=RuntimeError("read boom"))

    r = client.post(
        "/api/read-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "key": "K"},
    )
    assert r.status_code == 400


def test_read_watermark_success_returns_200(mocker, client, auth_headers, tmp_path):
    """
    Validates success branch in read-watermark:
    WMUtils.read_watermark returns secret -> 200 and json contains secret.
    """
    app = client.application
    app.config["STORAGE_DIR"] = tmp_path

    rel = "files/u/doc.pdf"
    _write_min_pdf(tmp_path / rel)
    row = _mk_select_row(doc_id=1, name="doc.pdf", path=rel)

    eng, _ = _mk_engine_for_select(row)
    mocker.patch("server.src.server.get_engine", return_value=eng)
    mocker.patch("server.src.server.WMUtils.read_watermark", return_value="SECRET")

    r = client.post(
        "/api/read-watermark/1",
        headers=auth_headers,
        json={"method": "fake-ok", "key": "K"},
    )
    assert r.status_code == 200
    assert r.get_json()["secret"] == "SECRET"
