"""fetch_geodata: a download is installed only when its sha256 and size match the record (never a partial or a
replaced file), present files are not fetched again, --check reports without downloading, and every source
record is complete. Uses file:// URLs in a temporary folder; nothing is downloaded."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fetch_geodata as F


class Fetch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-fetch-tests-')
        t = Path(self._tmp.name)
        self.root, self.pub = t / 'root', t / 'pub'
        self.pub.mkdir()
        (self.pub / 'a.nc').write_bytes(b'elevation' * 1000)
        with zipfile.ZipFile(self.pub / 'b.zip', 'w') as z:
            z.writestr('b.tif', b'relief')
        self.entries = {
            'a': self.entry('a.nc', 'work/geodata/a.nc'),
            'b': dict(self.entry('b.zip', 'work/geodata/ne1/b.zip'), optional='preview', extract_to='work/geodata/ne1'),
        }
        self.sources = t / 'sources.json'
        self.write()
        self.calls = []

    def tearDown(self):
        self._tmp.cleanup()

    def entry(self, name, path):
        d = (self.pub / name).read_bytes()
        return dict(path=path, url=(self.pub / name).as_uri(), bytes=len(d), sha256=hashlib.sha256(d).hexdigest())

    def write(self):
        self.sources.write_text(json.dumps(dict(format=1, files=self.entries)), encoding='utf-8')

    def opener(self, url):
        self.calls.append(url)
        return F.urllib.request.urlopen(url)

    def run_main(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return F.main(list(argv), root=self.root, sources=self.sources, opener=self.opener)

    def test_fetch_check_and_no_refetch(self):
        self.assertEqual(self.run_main('--check'), 1)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.run_main(), 0)
        self.assertEqual((self.root / 'work/geodata/a.nc').read_bytes(), (self.pub / 'a.nc').read_bytes())
        self.assertFalse((self.root / 'work/geodata/ne1/b.zip').exists())        # optional: not by default
        self.assertEqual(self.run_main('--check'), 0)
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(len(self.calls), 1)                                     # present file not fetched again
        self.assertEqual(self.run_main('--optional'), 0)
        self.assertEqual((self.root / 'work/geodata/ne1/b.tif').read_bytes(), b'relief')

    def test_replaced_file_is_not_installed(self):
        self.entries['a']['sha256'] = '0' * 64
        self.write()
        with self.assertRaises(SystemExit) as e:
            self.run_main()
        self.assertIn('differs from the recorded', str(e.exception))
        self.assertEqual(list((self.root / 'work/geodata').iterdir()), [])       # no file, no .part left

    def test_existing_wrong_file_is_left_alone(self):
        p = self.root / 'work/geodata/a.nc'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'old')
        with self.assertRaises(SystemExit):
            self.run_main()
        self.assertEqual(p.read_bytes(), b'old')
        self.assertEqual(self.calls, [])

    def test_source_records_complete(self):
        rec = json.loads(F.SOURCES.read_text(encoding='utf-8'))['files']
        self.assertEqual(set(rec), {'etopo', 'ecoregions', 'ne_rivers', 'ne_lakes', 'ne_playas', 'ne1_relief'})
        for name, e in rec.items():
            for k in ('path', 'url', 'bytes', 'sha256', 'dataset', 'licence'):
                self.assertIn(k, e, name)
            self.assertTrue(e['url'].startswith('https://'), name)
            self.assertRegex(e['sha256'], r'^[0-9a-f]{64}$')
            self.assertTrue(e['path'].startswith('work/geodata/'), name)
        layout = json.loads((F.SOURCES.parent / 'map-layout/geodata.json').read_text(encoding='utf-8'))['files']
        for k in ('etopo', 'ne_rivers'):                                         # same file as the bake checks
            self.assertEqual(rec[k]['sha256'], layout[k]['sha256'])
            self.assertEqual(rec[k]['path'], layout[k]['path'])


if __name__ == '__main__':
    unittest.main()
