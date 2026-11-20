# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Offline MONO dispatch equivalence checks; run directly without GPU packages."""

import unittest


WORKERS = 256
OLD_DISPATCH = [(x, y) for y in range(16) for x in range(16)]


def gemm_tile(task, m_tiles, n_tiles, batches):
    strip = 0
    if batches == 1:
        if m_tiles == 2 and n_tiles % 16 == 0:
            strip = 16
        elif m_tiles == 3 and n_tiles % 4 == 0:
            strip = 4
    if strip:
        stripe = task // strip
        return stripe % m_tiles, (stripe // m_tiles) * strip + task % strip, 0
    mn = task // m_tiles
    return task % m_tiles, mn % n_tiles, mn // n_tiles


def batched_tasks(worker, count, batch):
    return [
        first + slot * WORKERS
        for first in range(worker, count, WORKERS * batch)
        for slot in range(batch)
        if first + slot * WORKERS < count
    ]


class LinearWorkerMappingTest(unittest.TestCase):
    def test_dispatch_order(self):
        self.assertEqual([x + 16 * y for x, y in OLD_DISPATCH], list(range(WORKERS)))

    def test_a01_split32_and_cluster(self):
        blocks = []
        for w, (x, y) in enumerate(OLD_DISPATCH):
            task = x + 16 * y
            old = (task % 8) * 32 + task // 8
            new = (w % 8) * 32 + w // 8
            self.assertEqual(old, new)
            blocks.append(new)
        self.assertEqual(sorted(blocks), list(range(WORKERS)))
        for x in range(16):
            for y0 in range(0, 16, 4):
                tasks = [x + 16 * y for y in range(y0, y0 + 4)]
                self.assertEqual(len({task % 8 for task in tasks}), 1)
                self.assertEqual(len({task // 8 for task in tasks}), 4)

    def test_a02_a04_a08_a10_gemm(self):
        for op in ("A02", "A04", "A08", "A10"):
            for mt, nt, batches in ((2, 128, 1), (3, 8, 1), (1, 384, 1),
                                    (4, 16, 1), (3, 47, 2)):
                with self.subTest(op=op, m_tiles=mt, n_tiles=nt, batches=batches):
                    count = mt * nt * batches
                    tiles = []
                    for w, (x, y) in enumerate(OLD_DISPATCH):
                        old = [gemm_tile(t, mt, nt, batches)
                               for t in range(x + 16 * y, count, WORKERS)]
                        new = [gemm_tile(t, mt, nt, batches)
                               for t in range(w, count, WORKERS)]
                        self.assertEqual(old, new)
                        tiles.extend(new)
                    expected = {(m, n, b) for b in range(batches)
                                for n in range(nt) for m in range(mt)}
                    self.assertEqual(len(tiles), count)
                    self.assertEqual(set(tiles), expected)

    def test_a05_query_and_kv_transpose(self):
        for tokens in (0, 1, 16, 96, 384, 513):
            for slots in (16, 17):
                count = tokens * slots
                q_tiles = (tokens * 16 + 7) // 8
                kv_tiles = (tokens + 3) // 4 if slots > 16 else 0
                tiles = []
                for w, (x, y) in enumerate(OLD_DISPATCH):
                    self.assertEqual(list(range(x + 16 * y, count, WORKERS)),
                                     list(range(w, count, WORKERS)))
                    old = [y + 16 * x + i * WORKERS
                           for i in range((q_tiles + kv_tiles + 255) // WORKERS)]
                    new = [(w >> 4) + 16 * (w & 15) + i * WORKERS
                           for i in range((q_tiles + kv_tiles + 255) // WORKERS)]
                    self.assertEqual(old, new)
                    tiles.extend(t for t in new if t < q_tiles + kv_tiles)
                self.assertEqual(sorted(tiles), list(range(q_tiles + kv_tiles)))

    def test_a07_wave_tasks(self):
        for tokens in (0, 1, 16, 96, 257, 384, 513):
            tasks = []
            for w, (x, y) in enumerate(OLD_DISPATCH):
                worker = x + 16 * y
                for wave in range(4):
                    if tokens == 96:
                        old = [worker * 3 + wave] if wave < 3 else []
                        new = [w * 3 + wave] if wave < 3 else []
                    else:
                        old = list(range(worker + wave * WORKERS, tokens * 8, 1024))
                        new = list(range(w + wave * WORKERS, tokens * 8, 1024))
                    self.assertEqual(old, new)
                    tasks.extend(new)
            self.assertEqual(sorted(tasks), list(range(tokens * 8)))

    def test_a03_a09_task_batching(self):
        for op in ("A03", "A09-FP8", "A09-QAT"):
            for count in (0, 1, 96, 192, 255, 256, 257, 512, 513, 767, 768, 1026):
                for batch in (1, 2, 4):
                    with self.subTest(op=op, count=count, batch=batch):
                        tasks = []
                        for w, (x, y) in enumerate(OLD_DISPATCH):
                            # The actual previous A03/A09 grid was (1, 256).
                            old_y = x + 16 * y
                            old = batched_tasks(old_y, count, batch)
                            new = batched_tasks(w, count, batch)
                            self.assertEqual(old, new)
                            self.assertEqual(new, list(range(w, count, WORKERS)))
                            tasks.extend(new)
                        self.assertEqual(sorted(tasks), list(range(count)))


if __name__ == "__main__":
    unittest.main()
