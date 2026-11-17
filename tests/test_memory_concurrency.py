"""MemoryStore under concurrent access.

The HTTP server is threaded, so with more than one client in flight the store is
mutated and searched at the same time. Before the lock, an eviction rebuild could
re-point the TF-IDF index under a concurrent search, so a recall could return an
episode that was no longer in the list (or raise IndexError).

These tests are probabilistic by nature; they are written to fail loudly and
often on the pre-lock implementation (a missing lock shows up as wrong episodes
or an IndexError within a few thousand interleavings) and never on a correct one.
"""

from __future__ import annotations

import threading
import unittest

from apex.memory.store import Episode, MemoryStore


class TestConcurrentWrites(unittest.TestCase):
    def test_parallel_episodes_respect_limit(self):
        store = MemoryStore(episodic_limit=50)
        errors: list[BaseException] = []

        def worker(worker_id: int):
            try:
                for i in range(100):
                    store.remember_episode(Episode(task=f"w{worker_id} task {i}", output=f"out {i}"))
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(errors, [])
        self.assertLessEqual(len(store.episodic), 50)
        self.assertEqual(store.stats()["episodes"], len(store.episodic))

    def test_recall_during_eviction_returns_live_episodes(self):
        store = MemoryStore(episodic_limit=25)
        stop = threading.Event()
        problems: list[str] = []
        seen: set[int] = set()

        def writer():
            i = 0
            while not stop.is_set():
                store.remember_episode(Episode(task=f"deploy pipeline {i}", output="blue-green"))
                i += 1

        def reader():
            while not stop.is_set():
                try:
                    hits = store.recall_episodes("deploy pipeline", top_k=5, min_score=0.05)
                except BaseException as exc:  # noqa: BLE001
                    problems.append(f"{type(exc).__name__}: {exc}")
                    return
                for episode in hits:
                    if episode not in store.episodic:
                        problems.append("recalled an episode that is no longer stored")
                        return
                seen.add(len(hits))

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for thread in threads:
            thread.start()
        threading.Event().wait(1.0)
        stop.set()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(problems, [])
        self.assertLessEqual(len(store.episodic), 25)

    def test_parallel_facts_are_deduplicated(self):
        store = MemoryStore(semantic_limit=500)
        errors: list[BaseException] = []

        def worker():
            try:
                for _ in range(50):
                    store.remember_fact("Staging deploys use blue-green rollout")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        self.assertEqual(errors, [])
        # Last-write-wins is fine; duplicated identical facts are not. A couple of
        # duplicates can survive a genuine race between check and append, so allow
        # one per thread as the documented worst case rather than pretending the
        # check-and-insert is atomic across threads.
        self.assertLessEqual(len(store.facts), 6)
        self.assertGreaterEqual(len(store.facts), 1)

    def test_snapshot_during_writes_is_consistent(self):
        store = MemoryStore(episodic_limit=100)
        stop = threading.Event()
        problems: list[str] = []

        def writer():
            i = 0
            while not stop.is_set():
                store.remember_episode(Episode(task=f"t{i}", output="o"))
                i += 1

        def snapshotter():
            while not stop.is_set():
                data = store.snapshot()
                if len(data["episodic"]) > 100:
                    problems.append("snapshot exceeded the episodic limit")
                    return

        threads = [threading.Thread(target=writer), threading.Thread(target=snapshotter)]
        for thread in threads:
            thread.start()
        threading.Event().wait(0.5)
        stop.set()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(problems, [])

    def test_procedure_upgrade_under_contention(self):
        store = MemoryStore()
        scores = list(range(1, 60))

        def worker(batch: list[int]):
            for score in batch:
                store.remember_procedure("code", "react", "m", score / 100)

        threads = [
            threading.Thread(target=worker, args=(scores[i::4],)) for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        best = store.best_procedure("code")
        self.assertIsNotNone(best)
        self.assertAlmostEqual(best["score"], 0.59)


if __name__ == "__main__":
    unittest.main()
