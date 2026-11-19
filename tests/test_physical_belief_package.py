"""Dependency-free validation of the isolated historical civilian package."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'algorithms/physical_belief_rocket_landing'
spec = importlib.util.spec_from_file_location('pbn_snapshot_test', ROOT / 'scripts/snapshot.py')
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


class PhysicalBeliefPackageTests(unittest.TestCase):
    def test_archive_retained_payloads_and_scope(self):
        record = json.loads((ROOT / 'civilian_integration.json').read_text())['archive']
        with tarfile.open(ROOT / 'vendor/benchmark_source.tar.gz') as stream:
            members = snapshot.validate_members(stream)
            actual = {}
            for m in members:
                self.assertNotIn('src/original_pyflyt/pz_envs', m.name)
                self.assertNotIn('missile', m.name.lower())
                if m.isfile():
                    actual[m.name] = hashlib.sha256(stream.extractfile(m).read()).hexdigest()
        self.assertEqual(actual, record['retained_file_sha256'])

    def test_reject_unsafe_members(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                           ('safe/link', tarfile.SYMTYPE), ('device', tarfile.CHRTYPE)]:
            with self.subTest(name=name):
                data = io.BytesIO()
                with tarfile.open(fileobj=data, mode='w') as stream:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    stream.addfile(member)
                data.seek(0)
                with tarfile.open(fileobj=data) as stream:
                    with self.assertRaises(RuntimeError):
                        snapshot.validate_members(stream)

    def test_prepare_and_verify(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            snapshot.prepare_snapshot(out)
            self.assertTrue(snapshot.verify_snapshot(out))

    def test_pinned_sources_and_checkpoint_hashes(self):
        provenance = json.loads((ROOT / 'provenance.json').read_text())
        for name, digest in provenance['actor_source_sha256'].items():
            self.assertEqual(snapshot.sha256(ROOT / 'core' / name), digest)
        for name, digest in provenance['artifact_sha256'].items():
            self.assertEqual(snapshot.sha256(ROOT / name), digest)


if __name__ == '__main__':
    unittest.main()
