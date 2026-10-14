import copy
import json
from pathlib import Path
import unittest
from tool import analyze

class TestTool(unittest.TestCase):
    def setUp(self):
        self.data=copy.deepcopy(json.loads((Path(__file__).resolve().parents[1]/'examples/input.json').read_text()))

    def test_exact_cross_split_chemistry_collision(self):
        result = analyze(self.data)
        self.assertEqual(len(result['cross_split_pairs']),2)
        self.assertEqual(len(result['chemistry_or_label_collisions']),1)
        self.assertFalse(result['cross_split_pairs'][0]['same_chemistry'])
    def test_exact_threshold_excludes_mutation(self):
        self.data['similarity_threshold']=1
        self.assertEqual(len(analyze(self.data)['cross_split_pairs']),1)
    def test_edit_metric(self):
        from tool import edit_similarity
        self.assertAlmostEqual(edit_similarity('ACDE','ACD'),.75)
        self.assertEqual(edit_similarity('AAAA','DDDD'),0)
    def test_invalid_residues(self):
        self.data['rows'][0]['sequence']='ABC'
        with self.assertRaises(ValueError): analyze(self.data)
    def test_duplicate_ids(self):
        self.data['rows'][1]['id']=self.data['rows'][0]['id']
        with self.assertRaises(ValueError): analyze(self.data)
    def test_same_split_is_not_leakage(self):
        for r in self.data['rows']: r['split']='train'
        self.assertEqual(analyze(self.data)['cross_split_pairs'],[])

if __name__ == '__main__':
    unittest.main()
