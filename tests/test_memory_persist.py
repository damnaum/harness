"""Durable memory tests — persistence, de-duplication, eviction integrity.

The eviction case is a regression test for a real bug: the TF-IDF index maps
document position to list position, so trimming the episode list without
re-indexing silently returned the *wrong* episode.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from apex.memory.store import Episode, MemoryStore


class TestEpisodic(unittest.TestCase):
    def test_recall_finds_relevant_episode(self):
        store = MemoryStore()
        store.remember_episode(Episode(task="deploy the kubernetes cluster", output="used helm"))
        store.remember_episode(Episode(task="bake a sourdough loaf", output="used a levain"))
        hits = store.recall_episodes("how do we deploy kubernetes")
        self.assertTrue(hits)
        self.assertIn("kubernetes", hits[0].task)

    def test_identical_task_is_not_recalled(self):
        store = MemoryStore()
        store.remember_episode(Episode(task="what is the capital of France", output="Paris"))
        self.assertEqual(store.recall_episodes("what is the capital of France"), [])

    def test_eviction_keeps_index_aligned(self):
        store = MemoryStore(episodic_limit=3)
        for i in range(6):
            store.remember_episode(Episode(task=f"topic{i} question about widget{i}", output=f"answer{i}"))
        self.assertEqual(len(store.episodic), 3)
        # Query differs from the stored task so the self-echo filter does not
        # hide the very episode under test.
        hits = store.recall_episodes("how should widget5 be handled")
        self.assertTrue(hits)
        self.assertEqual(hits[0].task, "topic5 question about widget5")

    def test_low_similarity_filtered(self):
        store = MemoryStore()
        store.remember_episode(Episode(task="kubernetes deployment", output="helm"))
        self.assertEqual(store.recall_episodes("sourdough bread recipe"), [])


class TestSemantic(unittest.TestCase):
    def test_fact_deduplication(self):
        store = MemoryStore()
        store.remember_fact("The staging cluster is in eu-west-1")
        store.remember_fact("  the staging   cluster is in EU-WEST-1 ")
        self.assertEqual(len(store.facts), 1)

    def test_fact_recall(self):
        store = MemoryStore()
        store.remember_fact("Deploys use blue-green rollout behind a feature flag")
        hits = store.recall_facts("how do deploys work")
        self.assertEqual(len(hits), 1)
        self.assertIn("blue-green", hits[0].text)

    def test_fact_limit(self):
        store = MemoryStore(semantic_limit=2)
        for i in range(5):
            store.remember_fact(f"distinct fact number {i} about subject{i}")
        self.assertEqual(len(store.facts), 2)


class TestProcedural(unittest.TestCase):
    def test_keeps_best_score(self):
        store = MemoryStore()
        store.remember_procedure("code", "react", "gpt-5.2", 0.4)
        store.remember_procedure("code", "reflexion", "claude-opus-4.5", 0.9)
        store.remember_procedure("code", "direct", "gpt-5-mini", 0.2)
        best = store.best_procedure("code")
        self.assertEqual(best["strategy"], "reflexion")
        self.assertEqual(best["score"], 0.9)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "nested" / "memory.json"

    def tearDown(self):
        for root, dirs, files in os.walk(self.tmp, topdown=False):
            for name in files:
                os.unlink(Path(root) / name)
            for name in dirs:
                os.rmdir(Path(root) / name)
        os.rmdir(self.tmp)

    def test_save_creates_parent_dirs(self):
        store = MemoryStore()
        store.remember_fact("a durable fact")
        store.save(self.path)
        self.assertTrue(self.path.exists())

    def test_roundtrip_preserves_everything(self):
        store = MemoryStore()
        store.remember_fact("pricing is in EUR")
        store.remember_episode(Episode(task="quote a customer", output="sent quote", score=0.8))
        store.remember_procedure("writing", "cot", "claude-sonnet-4.5", 0.7)
        store.save(self.path)

        restored = MemoryStore()
        self.assertTrue(restored.load(self.path))
        self.assertEqual([f.text for f in restored.facts], ["pricing is in EUR"])
        self.assertEqual(restored.episodic[0].task, "quote a customer")
        self.assertEqual(restored.best_procedure("writing")["model"], "claude-sonnet-4.5")

    def test_restored_index_is_searchable(self):
        store = MemoryStore()
        store.remember_fact("the on-call rotation starts on Monday")
        store.save(self.path)
        restored = MemoryStore()
        restored.load(self.path)
        self.assertTrue(restored.recall_facts("when does the on-call rotation start"))

    def test_atomic_write_leaves_no_tmp(self):
        store = MemoryStore()
        store.save(self.path)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["memory.json"])

    def test_corrupt_file_is_survivable(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json", encoding="utf-8")
        store = MemoryStore()
        self.assertFalse(store.load(self.path))
        self.assertEqual(store.stats()["facts"], 0)

    def test_missing_file_is_survivable(self):
        self.assertFalse(MemoryStore().load(Path(self.tmp) / "absent.json"))

    def test_snapshot_is_json_serialisable(self):
        store = MemoryStore()
        store.remember_episode(Episode(task="t", output="o"))
        json.dumps(store.snapshot())


if __name__ == "__main__":
    unittest.main()
