"""extract_vanilla_reference: a file is taken only from its named vanilla pack and only with the recorded hash
(another pack holding the same path is ignored; a changed game stops the run and writes nothing); present files
are left as they are; --check only reports. Synthetic PFH5 packs in a temporary folder."""
import contextlib
import hashlib
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import extract_vanilla_reference as E

KEY = 'terrain/battles/3k_main_map/battle_locations_map.bin'


def pfh5(entries):
    """A minimal uncompressed PFH5 pack: header, index (size, flag, name), data."""
    index = b''.join(struct.pack('<I', len(d)) + b'\0' + p.encode() + b'\0' for p, d in entries.items())
    return b'PFH5' + struct.pack('<6I', 0, 0, 0, len(entries), len(index), 0) + index + b''.join(entries.values())


class Extract(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-vanilla-ref-tests-')
        t = Path(self._tmp.name)
        self.data, self.out = t / 'data', t / 'out'
        self.data.mkdir()
        self.vanilla = b'FASTBIN0 vanilla bytes'
        (self.data / 'fast.pack').write_bytes(pfh5({'other/file.txt': b'x', KEY: self.vanilla}))
        (self.data / 'mod.pack').write_bytes(pfh5({KEY: b'a mod edit'}))
        self.files = {KEY: ('fast.pack', len(self.vanilla), hashlib.sha256(self.vanilla).hexdigest())}

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, *argv, files=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return E.main(list(argv), game_data=self.data, out=self.out, files=files or self.files)

    def test_extract_from_named_pack_then_check(self):
        self.assertEqual(self.run_main('--check'), 1)
        self.assertFalse((self.out / KEY).exists())
        self.assertEqual(self.run_main(), 0)
        self.assertEqual((self.out / KEY).read_bytes(), self.vanilla)
        self.assertEqual(self.run_main('--check'), 0)

    def test_changed_game_stops(self):
        files = {KEY: ('fast.pack', len(self.vanilla), '0' * 64)}
        with self.assertRaises(SystemExit) as e:
            self.run_main(files=files)
        self.assertIn('the game was updated', str(e.exception))
        self.assertFalse(self.out.exists())

    def test_present_file_kept(self):
        self.run_main()
        before = (self.out / KEY).stat().st_mtime_ns
        self.run_main()
        self.assertEqual((self.out / KEY).stat().st_mtime_ns, before)

    def test_recorded_entry_matches_this_machine(self):
        """The recorded hash is the vanilla file of the installed game (skipped without a configured game)."""
        pack, size, sha = E.FILES[KEY]
        p = E.tw3k_paths.GAME_DATA / pack
        if not p.is_file():
            self.skipTest('no game configured')
        data = E.read_entry(p, KEY)
        self.assertEqual((len(data), hashlib.sha256(data).hexdigest()), (size, sha))


if __name__ == '__main__':
    unittest.main()
