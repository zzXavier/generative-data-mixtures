import unittest
from src.curation import ARMS,load,score_threshold,select

class SelectionTests(unittest.TestCase):
    def test_budget_grouping_and_split(self):
        _,_,splits,rows,_ = load(5701)
        total=sum(len(s) for s in splits.values())
        self.assertEqual(len(set().union(*(set(x) for x in splits.values()))),total)
        for arm in ARMS:
            chosen=select(rows,arm,120,5701,.68)
            self.assertEqual(len(chosen),120)
            self.assertEqual(sum(r["polarity"] for r in chosen),int(120*ARMS[arm][1]))
            if "dedup" in arm or "combined" in arm:
                self.assertEqual(len({r["signature"] for r in chosen}),120)
    def test_inclusive_score_cut(self):
        self.assertEqual(score_threshold([0.,.2,.2,.9],.5),.2)
