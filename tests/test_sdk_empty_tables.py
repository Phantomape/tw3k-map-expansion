"""Empty compiler containers use official SDK formats, never reference gameplay."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sdk_empty_tables as sdk


def container(version=None, rows=0, payload=b""):
    guid = str(uuid.UUID("11111111-2222-3333-4444-555555555555")).encode("utf-16le")
    block = b"" if version is None else b"\xfc\xfd\xfe\xff" + struct.pack("<I", version)
    return b"\xfd\xfe\xfc\xff" + struct.pack("<H", len(guid) // 2) + guid + block + b"\x01" + struct.pack("<I", rows) + payload


class OfficialEmptyTables(unittest.TestCase):
    def test_versioned_and_unversioned_official_formats(self):
        self.assertEqual(sdk.empty_table_format(container()), b"")
        for version in (1, 4):
            self.assertEqual(sdk.empty_table_format(container(version)), b"\xfc\xfd\xfe\xff" + struct.pack("<I", version))

    def test_nonempty_official_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contains records"):
            sdk.empty_table_format(container(rows=59, payload=b"rows"))

    def test_trailing_payload_is_rejected_even_with_zero_count(self):
        with self.assertRaisesRegex(ValueError, "trailing payload"):
            sdk.empty_table_format(container(payload=b"undeclared rows"))

    def test_corrupt_headers_fail_closed(self):
        original = container(4)
        bad_guid = bytearray(original); bad_guid[6:8] = b"\x00\xd8"
        bad_flag = bytearray(original); bad_flag[-5] = 0
        bad_length = bytearray(original); bad_length[4:6] = struct.pack("<H", 65535)
        for packed in (b"", b"not db", bytes(bad_guid), bytes(bad_flag), bytes(bad_length), original[:-1], original[:82]):
            with self.subTest(packed=packed[:6]):
                with self.assertRaises(ValueError):
                    sdk.empty_table_format(packed)

    def test_unknown_table_uses_its_explicit_official_export(self):
        with tempfile.TemporaryDirectory(prefix="tw3k-sdk-empty-") as tmp:
            root = Path(tmp); source = root / "start_pos_future_internal_tables" / "!tw3k_exp"
            source.parent.mkdir(); original = container(17); source.write_bytes(original)
            made, report = sdk.from_official_export("start_pos_future_internal", root)
            self.assertEqual(sdk.empty_table_format(made), sdk.empty_table_format(original))
            self.assertNotEqual(made, original)
            self.assertEqual(report["rows"], 0)
            self.assertEqual(report["official_sdk_source"], str(source))
            self.assertEqual(len(report["official_sdk_source_sha256"]), 64)
            self.assertEqual(made, sdk.from_official_export("start_pos_future_internal", root)[0])

    def test_missing_official_source_does_not_fall_back(self):
        with tempfile.TemporaryDirectory(prefix="tw3k-sdk-empty-") as tmp:
            root = Path(tmp)
            # A reference-like sibling cannot satisfy the explicit SDK input.
            decoy = root / "reference" / "some-mod" / "db" / "start_pos_missing_tables"
            decoy.mkdir(parents=True); (decoy / "data_3k_main_campaign_map").write_bytes(container())
            with self.assertRaisesRegex(ValueError, "No official SDK zero-row export"):
                sdk.from_official_export("start_pos_missing", root / "official")

    def test_table_name_cannot_escape_official_directory(self):
        for table in ("../reference", "a/b", "a\\b", "", "UPPER"):
            with self.subTest(table=table):
                with self.assertRaisesRegex(ValueError, "Invalid SDK table name"):
                    sdk.from_official_export(table, Path("unused"))


class ExportPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import export_build_tables
        cls.exporter = export_build_tables

    def fixture(self, root, row=""):
        kit = root / "kit"
        xml = kit / "raw_data/db/start_pos_unknown.xml"
        xml.parent.mkdir(parents=True)
        xml.write_text("<dataroot>" + row + "</dataroot>", encoding="utf-8")
        (root / "analysis").mkdir()
        (root / "analysis/standalone-prototype.json").write_text('{"changed_rows": {}}', encoding="utf-8")
        return kit

    def invoke(self, root, kit, args=()):
        ex = self.exporter
        with patch.object(ex, "ROOT", root), patch.object(ex, "KIT", kit), patch.object(ex, "OUT", root / "out"), \
                patch.object(ex, "definitions", return_value={}), patch.object(ex.buddhist_temple_db, "ENABLED", True), \
                patch.object(sys, "argv", ["export_build_tables.py", *args]):
            ex.main()

    def test_default_sdk_directory_follows_redirected_workspace_root(self):
        with tempfile.TemporaryDirectory(prefix="tw3k-sdk-preflight-") as tmp:
            root = Path(tmp); kit = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "No official SDK zero-row export") as caught:
                self.invoke(root, kit)
            self.assertIn(str(root / "work/native-db/start_pos_unknown_tables/!tw3k_exp"), str(caught.exception))
            self.assertFalse((root / "out").exists())
            self.assertFalse((root / "work").exists())

    def test_explicit_sdk_directory_is_used_and_bad_source_writes_nothing(self):
        with tempfile.TemporaryDirectory(prefix="tw3k-sdk-preflight-") as tmp:
            root = Path(tmp); kit = self.fixture(root); source = root / "official/start_pos_unknown_tables/!tw3k_exp"
            source.parent.mkdir(parents=True); source.write_bytes(container(rows=1, payload=b"record"))
            with self.assertRaisesRegex(ValueError, "contains records"):
                self.invoke(root, kit, ["--official-empty-db", str(root / "official")])
            self.assertFalse((root / "out").exists())
            self.assertFalse((root / "work").exists())

    def test_nonempty_unknown_xml_is_not_silently_replaced_by_empty_header(self):
        with tempfile.TemporaryDirectory(prefix="tw3k-sdk-preflight-") as tmp:
            root = Path(tmp); kit = self.fixture(root, "<start_pos_unknown><id>1</id></start_pos_unknown>")
            with self.assertRaisesRegex(ValueError, "No schema for nonempty table start_pos_unknown"):
                self.invoke(root, kit)
            self.assertFalse((root / "out").exists())


class NativeExportRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import all_directions_build
        cls.builder = all_directions_build

    def test_bootstrap_reads_raw_official_xml_without_split_or_reference_inputs(self):
        with tempfile.TemporaryDirectory(prefix="tw3k-sdk-bootstrap-") as tmp:
            root = Path(tmp); kit = root / "kit"; converter = SimpleNamespace(main=Mock())
            with patch.object(self.builder, "A", root), patch.object(self.builder, "KIT", kit), \
                    patch.dict(sys.modules, {"native_db_export": converter}), patch.object(sys, "argv", []):
                self.builder.official_headers()
                self.assertEqual(converter.ROOT, root)
                self.assertEqual(converter.BIN, kit / "binaries")
                self.assertEqual(sys.argv, ["native_db_export.py", "--all"])
                converter.main.assert_called_once_with()
            self.assertEqual(list(root.iterdir()), [])

    def test_regular_native_export_keeps_campaign_split(self):
        with tempfile.TemporaryDirectory(prefix="tw3k-sdk-bootstrap-") as tmp:
            root = Path(tmp); converter = SimpleNamespace(main=Mock())
            with patch.object(self.builder, "A", root), patch.object(self.builder, "KIT", root / "kit"), \
                    patch.dict(sys.modules, {"native_db_export": converter}), patch.object(sys, "argv", []):
                self.builder.native()
                self.assertEqual(sys.argv, ["native_db_export.py", "--all", "--split"])
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
