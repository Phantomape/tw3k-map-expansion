"""tw3k_paths: environment variable beats tw3k.local.json beats the default; a worktree and a scripts copy outside
any checkout both find the main checkout's config (so their pipeline locks stay shared); an unconfigured game path
imports as UNSET-<variable> and require() names what to set. Temporary folders only; os.environ is restored."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import tw3k_paths as T


class Paths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='tw3k-paths-tests-')
        t = Path(self._tmp.name)
        self.main, self.wt, self.copy, self.appdata = t / 'TW3K', t / 'wt-x', t / 'copy', t / 'appdata'
        (self.main / '.git' / 'worktrees' / 'wt-x').mkdir(parents=True)
        self.wt.mkdir()
        (self.wt / '.git').write_text(f'gitdir: {(self.main / ".git/worktrees/wt-x").as_posix()}\n', encoding='utf-8')
        self.copy.mkdir()
        self.appdata.mkdir()
        self.env = mock.patch.dict(os.environ, {'APPDATA': str(self.appdata)})
        self.env.start()
        for k in T.KEYS.values():
            os.environ.pop(k, None)

    def tearDown(self):
        self.env.stop()
        self._tmp.cleanup()

    def config(self, where, **kv):
        (where / T.CONFIG).write_text(json.dumps(kv), encoding='utf-8')

    def test_main_checkout_of_a_worktree(self):
        self.assertEqual(T.main_checkout(self.wt), self.main)
        self.assertEqual(T.main_checkout(self.main), self.main)
        self.assertEqual(T.main_checkout(self.copy), self.copy)

    def test_order_env_config_default(self):
        game, env_game, rpfm = (str(self.main.parent / n) for n in ('game-from-config', 'g', 'r.exe'))
        self.config(self.main, game_dir=game)
        self.assertIsNone(T.resolve('game_dir', self.copy))
        self.assertEqual(T.resolve('game_dir', self.wt), Path(game))                     # via the main checkout
        self.assertEqual(T.resolve('game_dir', self.wt, env={'TW3K_GAME_DIR': env_game}), Path(env_game))
        self.config(self.wt, rpfm_cli=rpfm)                                               # own config wins, whole file
        self.assertEqual(T.resolve('rpfm_cli', self.wt), Path(rpfm))
        self.assertIsNone(T.resolve('game_dir', self.wt))

    def test_defaults(self):
        self.assertEqual(T.resolve('main_root', self.wt), self.main)
        self.assertEqual(T.resolve('scratch', self.wt), self.main.with_name('TW3K-scratch'))
        self.assertEqual(T.resolve('rpfm_schema', self.wt),
                         self.appdata / 'FrodoWazEre/rpfm/config/schemas/schema_3k.ron')
        self.assertEqual(T.resolve('ca_appdata', self.wt), self.appdata / 'The Creative Assembly/ThreeKingdoms')
        win = str(self.main.parent / 'Win')
        self.assertEqual(T.resolve('fonts', self.wt, env={'WINDIR': win}), Path(win, 'Fonts'))

    def test_copy_outside_checkout_uses_appdata_config(self):
        self.assertEqual(T.resolve('main_root', self.copy), self.copy)
        (self.appdata / 'tw3k').mkdir()
        self.config(self.appdata / 'tw3k', main_root=str(self.main))
        self.assertEqual(T.resolve('main_root', self.copy), self.main)

    def test_unset_and_require(self):
        with mock.patch.object(T, 'ROOT', self.copy):
            with mock.patch.object(T, 'resolve', lambda k, root=None: None):
                self.assertEqual(T.GAME_DATA, Path('UNSET-TW3K_GAME_DIR') / 'data')
                with self.assertRaisesRegex(RuntimeError, 'TW3K_GAME_DIR'):
                    T.require('game_dir')
        with self.assertRaises(AttributeError):
            T.NOT_A_PATH


_LOCAL = T.load_config()


@unittest.skipUnless('_expect' in _LOCAL, 'no "_expect" table in this machine\'s tw3k.local.json')
class LocalMachine(unittest.TestCase):
    """On a machine whose tw3k.local.json records "_expect" (the paths its scripts held before they were
    configurable), every path resolves to the same location with no environment variable set."""
    def test_same_as_before(self):
        env = {k: v for k, v in os.environ.items() if k not in T.KEYS.values()}
        with mock.patch.dict(os.environ, env, clear=True):
            for name, want in _LOCAL['_expect'].items():
                got = T.font(name.split(':', 1)[1]) if name.startswith('font:') else getattr(T, name)
                self.assertEqual(os.path.normcase(Path(got)), os.path.normcase(Path(want)), name)


if __name__ == '__main__':
    unittest.main()
